"""
IC-LoRA Full — the "correct" IC-LoRA implementation for Anima (Cosmos-Predict2 DiT).

This is an improved version of ic_lora.py that addresses all known gaps
between our original implementation and the reference LTX-2 / Sync-LoRA approach:

Fixes vs ic_lora.py:
1. Loss normalization: loss computed ONLY over target tokens, not diluted by T=2
   (ic_lora.py averaged over the entire output including reference, halving gradients)
2. Multiscale loss: proper squeeze(2) before avg_pool2d
   (ic_lora.py used flatten(0,1) which operated on wrong dimensions)
3. Condition dropout: enables classifier-free guidance at inference
   (ic_lora.py had no dropout — CFG was not possible)
4. Configurable concat order: [ref, target] (LTX-2 style) or [target, ref] (original)
   (ic_lora.py hardcoded [target, ref])

Architecture (unchanged from ic_lora.py — these were already correct):
- Asymmetric noise: reference stays CLEAN (t=0), target gets noise (t=sigma)
- Per-token timestep: shape (B, 2) — Anima's t_embedder produces genuinely
  per-temporal-position AdaLN modulation (verified in Block.forward)
- 3D RoPE: T=2 works correctly, each frame gets distinct temporal position
- Full bidirectional attention: no causal mask needed
- Standard PEFT LoRA

References:
- LTX-2 IC-LoRA: https://github.com/Lightricks/ltx-video-trainer
- Sync-LoRA: arXiv:2512.03013
- IC-LoRA paper: arXiv:2410.23775
"""

import torch
import torch.nn.functional as F

from models.cosmos_predict2 import CosmosPredict2Pipeline, get_lin_function, time_shift, _tokenize


class ICLoraFullPipeline(CosmosPredict2Pipeline):
    """IC-LoRA Full training pipeline for Anima.

    Differences from ICLoraPipeline (ic_lora.py):
    - Corrected loss normalization (no gradient dilution from T=2)
    - Condition dropout for classifier-free guidance training
    - Configurable concat order (ref_first matches LTX-2)
    - Fixed multiscale loss for T>1 outputs

    Config options under [ic_lora_full]:
        condition_dropout: float (default: 0.1)
            Probability of dropping condition during training.
            Enables classifier-free guidance at inference.
        ref_first: bool (default: true)
            If true: [clean_ref T=0, noisy_target T=1] (LTX-2 style)
            If false: [noisy_target T=0, clean_ref T=1] (original ic_lora.py style)
            LTX-2 puts reference first so it gets the lower temporal RoPE positions.
        condition_timestep: float (default: 0.0)
            Timestep for condition tokens. 0.0 = clean (standard).
    """

    def __init__(self, config):
        super().__init__(config)
        oc_config = config.get('ic_lora_full', {})
        self.condition_dropout = oc_config.get('condition_dropout', 0.1)
        self.ref_first = oc_config.get('ref_first', True)
        self.condition_timestep = oc_config.get('condition_timestep', 0.0)

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        if self.cache_text_embeddings:
            prompt_embeds_or_batch_encoding = (
                inputs['prompt_embeds'], inputs['attn_mask'],
                inputs['t5_input_ids'], inputs['t5_attn_mask'],
            )
        else:
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

        # Timestep sampling (standard flow matching)
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

        # Asymmetric noise: only target gets noise
        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents  # velocity target, shape (B, C, 1, H, W)

        if 'control_latents' in inputs:
            control_latents = inputs['control_latents'].float()

            # Condition dropout for classifier-free guidance
            # prepare_inputs is only called during training
            if self.condition_dropout > 0:
                drop_mask = torch.rand(bs, device=latents.device) < self.condition_dropout
                if drop_mask.any():
                    control_latents = control_latents.clone()
                    control_latents[drop_mask] = 0.0

            # Per-token timestep
            cond_t = torch.full_like(t, self.condition_timestep)

            if self.ref_first:
                # LTX-2 style: [clean_ref T=0, noisy_target T=1]
                # Reference gets lower temporal RoPE positions (t=0)
                # Target gets higher temporal RoPE positions (t=1)
                noisy_latents = torch.cat([control_latents, noisy_latents], dim=2)
                t = torch.stack([cond_t, t], dim=1)  # (B, 2): [0, sigma]
            else:
                # Original style: [noisy_target T=0, clean_ref T=1]
                noisy_latents = torch.cat([noisy_latents, control_latents], dim=2)
                t = torch.stack([t, cond_t], dim=1)  # (B, 2): [sigma, 0]
        else:
            # No condition — standard single-frame training (fallback)
            t = t.view(-1, 1)  # (B, 1)

        return (noisy_latents, t, *prompt_embeds_or_batch_encoding), (target, mask)

    def get_loss_fn(self):
        ref_first = self.ref_first

        def loss_fn(output, label):
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)

                # Slice output to target frame only.
                # output shape: (B, C, T_total, H, W) where T_total may be 2
                # target shape: (B, C, 1, H, W) — always the target frame
                target_T = target.shape[2]  # 1
                if output.shape[2] > target_T:
                    if ref_first:
                        # [ref T=0, target T=1] — target is the LAST frame
                        output = output[:, :, -target_T:, :, :]
                    else:
                        # [target T=0, ref T=1] — target is the FIRST frame
                        output = output[:, :, :target_T, :, :]

                # Loss computation — normalized by TARGET size only (not diluted by T=2)
                if 'pseudo_huber_c' in self.config:
                    c = self.config['pseudo_huber_c']
                    loss = torch.sqrt((output - target) ** 2 + c ** 2) - c
                else:
                    loss = F.mse_loss(output, target, reduction='none')

                if mask is not None and mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask

                # Normalize by target token count (not total output count)
                # This is the key fix: loss.mean() would dilute by T=2 since
                # .mean() divides by numel of the sliced tensor, but the model
                # processed T=2 tokens while we only compute loss on T=1.
                # After slicing, output is (B, C, 1, H, W) so .mean() is correct.
                loss = loss.mean()

                if weight := self.multiscale_loss_weight:
                    assert output.ndim == 5 and target.ndim == 5
                    # Squeeze temporal dim for 2D pooling
                    output_2d = output.squeeze(2)  # (B, C, H, W)
                    target_2d = target.squeeze(2)
                    for factor in [2, 4]:
                        output_2d = F.avg_pool2d(output_2d, 2)
                        target_2d = F.avg_pool2d(target_2d, 2)
                        if 'pseudo_huber_c' in self.config:
                            c = self.config['pseudo_huber_c']
                            ds_loss = torch.sqrt(
                                (output_2d - target_2d) ** 2 + c ** 2
                            ) - c
                        else:
                            ds_loss = F.mse_loss(
                                output_2d, target_2d, reduction='none'
                            )
                        loss = loss + weight * ds_loss.mean()

            return loss
        return loss_fn
