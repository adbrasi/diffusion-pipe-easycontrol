"""
IC-LoRA (In-Context LoRA) pipeline for Anima.

Extends the standard CosmosPredict2Pipeline with:
1. Asymmetric noise: reference frame stays CLEAN (t=0), target frame gets noise (t=sigma)
2. Per-token timestep: t_B_T has shape (B, 2) — t[:,0]=sigma, t[:,1]=0
3. Loss masking: loss computed ONLY on target frame (reference excluded)

Based on:
- IC-LoRA paper (Alibaba): https://github.com/ali-vilab/In-Context-LoRA
- LTX-2 IC-LoRA trainer: https://github.com/Lightricks/LTX-2
- Sync-LoRA / TIC-FT approaches for video DiTs
"""

import torch
import torch.nn.functional as F

from models.cosmos_predict2 import CosmosPredict2Pipeline, get_lin_function, time_shift


class ICLoraPipeline(CosmosPredict2Pipeline):
    """IC-LoRA training pipeline for Anima.

    Key difference from levzzz (standard temporal concat):
    - levzzz: both frames get noise, single timestep → model denoises both
    - IC-LoRA: only target gets noise, reference stays clean, per-token timestep
      → model learns to USE reference as context

    Key difference from EasyControl:
    - EasyControl: custom LoRA with binary masking, causal attention, separate condition stream
    - IC-LoRA: standard PEFT LoRA, full bidirectional attention, per-token timestep differentiation
    """

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        if self.cache_text_embeddings:
            prompt_embeds_or_batch_encoding = (
                inputs['prompt_embeds'], inputs['attn_mask'],
                inputs['t5_input_ids'], inputs['t5_attn_mask'],
            )
        else:
            from models.cosmos_predict2 import _tokenize
            captions = inputs['caption']
            batch_encoding = _tokenize(self.tokenizer, captions)
            t5_batch_encoding = _tokenize(self.t5_tokenizer, captions)
            prompt_embeds_or_batch_encoding = (
                batch_encoding.input_ids, batch_encoding.attention_mask,
                t5_batch_encoding.input_ids, t5_batch_encoding.attention_mask,
            )

        bs, channels, num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')
            mask = mask.unsqueeze(2)

        # Standard flow matching noise schedule
        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

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
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        # Noise the TARGET frame
        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        if 'control_latents' in inputs:
            control_latents = inputs['control_latents'].float()

            # IC-LoRA: reference frame stays CLEAN (no noise added)
            # Temporal concat: [noisy_target_T0, clean_reference_T1]
            noisy_latents = torch.cat([noisy_latents, control_latents], dim=2)

            # Per-token timestep: target gets sigma, reference gets 0
            # Shape (B, 2) — the model's t_embedder handles per-temporal-position
            t = torch.stack([t, torch.zeros_like(t)], dim=1)  # (B, 2)
        else:
            # No control — standard single-frame training (fallback)
            t = t.view(-1, 1)  # (B, 1)

        return (noisy_latents, t, *prompt_embeds_or_batch_encoding), (target, mask)

    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)

                # IC-LoRA: output has T=2 (target + reference prediction)
                # Loss ONLY on the target frame (T=0), exclude reference
                if output.shape[2] != target.shape[2]:
                    output = output[:, :, :target.shape[2], :, :]

                if 'pseudo_huber_c' in self.config:
                    c = self.config['pseudo_huber_c']
                    loss = torch.sqrt((output - target) ** 2 + c ** 2) - c
                else:
                    loss = F.mse_loss(output, target, reduction='none')

                if mask is not None and mask.numel() > 0:
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
                        if mask is not None and mask.numel() > 0:
                            ds_loss *= mask
                        loss = loss + weight * ds_loss.mean()

            return loss
        return loss_fn
