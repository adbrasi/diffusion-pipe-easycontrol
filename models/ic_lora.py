"""
IC-LoRA (In-Context LoRA) pipeline for Anima.

Extends the standard CosmosPredict2Pipeline with asymmetric noise:
- Target frame (T=0): noisy (standard flow matching)
- Reference frame (T=1): CLEAN (timestep=0, no noise added)

The model learns to use the clean reference frame as context for generation.
Loss is computed ONLY on the target frame (reference frame is excluded).

This is much simpler than EasyControl (~50 lines vs ~780 lines):
- No causal attention mask
- No binary-masked LoRA
- No separate residual stream for condition
- Standard PEFT LoRA (rank 32) on all attention layers

Based on:
- IC-LoRA paper (Alibaba): https://github.com/ali-vilab/In-Context-LoRA
- LTX-2 IC-LoRA trainer: https://github.com/Lightricks/LTX-2
- Sync-LoRA / TIC-FT approaches for video DiTs
"""

import torch
import torch.nn.functional as F

from models.cosmos_predict2 import CosmosPredict2Pipeline


class ICLoraPipeline(CosmosPredict2Pipeline):
    """IC-LoRA training pipeline for Anima.

    Key difference from levzzz (standard temporal concat):
    - levzzz: both frames get noise → model denoises both
    - IC-LoRA: only target gets noise, reference stays clean → model uses reference as context

    Key difference from EasyControl:
    - EasyControl: custom LoRA with binary masking, causal attention, separate condition stream
    - IC-LoRA: standard PEFT LoRA, full bidirectional attention, per-token noise differentiation
    """

    def prepare_inputs(self, latents, *prompt_embeds_or_batch_encoding, mask=torch.tensor([])):
        inputs = self.get_vae_fn_results()
        latents = inputs['latents'].float()
        bs = latents.shape[0]

        # Standard flow matching noise schedule
        timestep_sample_method = self.config.get('timestep_sample_method', 'logit_normal')
        timestep_quantile = self.config.get('timestep_quantile', None)

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            from models.cosmos_predict2 import get_lin_function, time_shift
            h, w = latents.shape[3], latents.shape[4]
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        # Standard noise for TARGET frame
        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents
        t = t.view(-1, 1)

        if 'control_latents' in inputs:
            control_latents = inputs['control_latents'].float()

            # IC-LoRA: reference frame stays CLEAN (no noise)
            # This is the key difference from levzzz (which adds noise to both)
            # Temporal concat: [noisy_target_T0, clean_reference_T1]
            noisy_latents = torch.cat([noisy_latents, control_latents], dim=2)

        return (noisy_latents, t, *prompt_embeds_or_batch_encoding), (target, mask)

    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)

                # IC-LoRA: output has T=2 (target + reference prediction)
                # Loss ONLY on the target frame (T=0), exclude reference frame
                if output.shape[2] != target.shape[2]:
                    output = output[:, :, :target.shape[2], :, :]

                if 'pseudo_huber_c' in self.config:
                    c = self.config['pseudo_huber_c']
                    loss = torch.sqrt((output - target) ** 2 + c ** 2) - c
                else:
                    loss = F.mse_loss(output, target, reduction='none')

                # Apply optional masking
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean()

                if weight := self.multiscale_loss_weight:
                    assert output.ndim == 5 and target.ndim == 5
                    for factor in [2, 4]:
                        output_ds = F.interpolate(
                            output.flatten(0, 1),
                            scale_factor=1 / factor,
                            mode='bilinear',
                        ).unflatten(0, output.shape[:2])
                        target_ds = F.interpolate(
                            target.flatten(0, 1),
                            scale_factor=1 / factor,
                            mode='bilinear',
                        ).unflatten(0, target.shape[:2])
                        if 'pseudo_huber_c' in self.config:
                            c = self.config['pseudo_huber_c']
                            ds_loss = torch.sqrt(
                                (output_ds - target_ds) ** 2 + c ** 2
                            ) - c
                        else:
                            ds_loss = F.mse_loss(
                                output_ds, target_ds, reduction='none'
                            )
                        if mask.numel() > 0:
                            ds_loss *= mask
                        loss = loss + weight * ds_loss.mean()

            return loss
        return loss_fn
