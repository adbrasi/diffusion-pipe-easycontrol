"""
IC-LoRA V2 pipeline for Anima.

Best of both worlds: inherits the corrected training pipeline from ic_lora_full
(proper loss normalization, condition dropout for CFG, ref_first LTX-2 order,
fixed multiscale loss) with focused LoRA targets (self_attn + mlp only).

Key improvements over ic_lora (V1):
- Excludes adaln_modulation from LoRA (Anima has internal adaln_lora — double-LoRA causes amplification)
- Excludes cross_attn from LoRA (text conditioning, not visual consistency)
- Condition dropout (enables classifier-free guidance at inference)
- Corrected loss normalization (no gradient dilution from T=2)
- Fixed multiscale loss for T>1 outputs
- ref_first=True by default (LTX-2 convention: [ref T=0, target T=1])
- Shifted logit-normal timestep sampler (LTX-2 style, sequence-length-adaptive)

References:
- LTX-2 IC-LoRA trainer: https://github.com/Lightricks/ltx-video-trainer
- Opus reviewer analysis of adaln_modulation double-LoRA issue
- Empirical evidence: skip_adaln=True at inference improves quality dramatically
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import peft

from models.ic_lora_full import ICLoraFullPipeline
from models.cosmos_predict2 import get_lin_function, time_shift, _tokenize
from utils.common import is_main_process


def _shifted_logit_normal_shift(num_tokens, min_tokens=1024, max_tokens=4096, min_shift=0.95, max_shift=2.05):
    """Compute the shift parameter for the logit-normal distribution based on sequence length.

    Linear interpolation of shift based on token count (LTX-2 approach):
    - Shorter sequences (fewer tokens) → lower shift → timesteps closer to uniform
    - Longer sequences (more tokens) → higher shift → timesteps biased toward higher noise

    Args:
        num_tokens: number of spatial tokens (T * H/patch * W/patch)
        min_tokens: sequence length that maps to min_shift
        max_tokens: sequence length that maps to max_shift
        min_shift: shift for short sequences (default 0.95, near-uniform)
        max_shift: shift for long sequences (default 2.05, high-noise bias)

    Returns:
        float: the shift value for the logit-normal mean
    """
    t = max(0.0, min(1.0, (num_tokens - min_tokens) / (max_tokens - min_tokens)))
    return min_shift + t * (max_shift - min_shift)


class ICLoraV2Pipeline(ICLoraFullPipeline):
    """IC-LoRA V2: focused LoRA targets + corrected training pipeline.

    Inherits from ic_lora_full (prepare_inputs, get_loss_fn, condition dropout, ref_first).
    Overrides:
    - configure_adapter: excludes adaln_modulation and cross_attn from LoRA
    - prepare_inputs: adds shifted logit-normal timestep sampler (LTX-2 style)

    Additional config options under [ic_lora_full]:
        shifted_logit_normal: bool (default: true)
            When true and timestep_sample_method == 'logit_normal', uses sequence-length-
            adaptive shift instead of a fixed mean of 0. Ignored for 'uniform' sampling.
        min_tokens: int (default: 1024)
            Sequence length (tokens) that maps to min_shift.
        max_tokens: int (default: 4096)
            Sequence length (tokens) that maps to max_shift.
        min_shift: float (default: 0.95)
            Shift (logit-normal mean) for short sequences.
        max_shift: float (default: 2.05)
            Shift (logit-normal mean) for long sequences.
    """

    def __init__(self, config):
        super().__init__(config)
        oc_config = config.get('ic_lora_full', {})
        self.shifted_logit_normal = oc_config.get('shifted_logit_normal', True)
        self.min_tokens = oc_config.get('min_tokens', 1024)
        self.max_tokens = oc_config.get('max_tokens', 4096)
        self.min_shift = oc_config.get('min_shift', 0.95)
        self.max_shift = oc_config.get('max_shift', 2.05)

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

        # Timestep sampling
        # Anima patch_spatial=2, patch_temporal=1 → spatial tokens per frame = (H/2) * (W/2)
        # For the adaptive shift we use the target-frame spatial token count (num_frames=1 target).
        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            # Compute adaptive shift based on spatial token count of the target frame
            # h, w are already latent dims (after VAE 8x downscale); patch_spatial=2
            num_tokens = num_frames * (h // 2) * (w // 2)

            if self.shifted_logit_normal:
                shift = _shifted_logit_normal_shift(
                    num_tokens,
                    min_tokens=self.min_tokens,
                    max_tokens=self.max_tokens,
                    min_shift=self.min_shift,
                    max_shift=self.max_shift,
                )
            else:
                shift = 0.0  # standard logit-normal (mean=0, no shift)

            dist = torch.distributions.normal.Normal(shift, 1.0)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError(f'Unsupported timestep_sample_method: {timestep_sample_method}')

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if static_shift := self.model_config.get('shift', None):
            t = (t * static_shift) / (1 + (static_shift - 1) * t)
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
            if self.condition_dropout > 0:
                drop_mask = torch.rand(bs, device=latents.device) < self.condition_dropout
                if drop_mask.any():
                    control_latents = control_latents.clone()
                    control_latents[drop_mask] = 0.0

            # Per-token timestep
            cond_t = torch.full_like(t, self.condition_timestep)

            if self.ref_first:
                # LTX-2 style: [clean_ref T=0, noisy_target T=1]
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

    def configure_adapter(self, adapter_config):
        """Apply LoRA ONLY to self_attn and mlp modules inside Block/TransformerBlock.

        Excluded:
        - adaln_modulation: Anima has internal adaln_lora (bottleneck dim=256).
          Adding PEFT LoRA on top causes double-LoRA amplification.
        - cross_attn: handles text conditioning, not visual consistency.
          Training it wastes LoRA capacity on text alignment that the base model already does well.
        """
        target_linear_modules = set()
        for name, module in self.transformer.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
                continue
            if name.startswith('llm_adapter'):
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    parts = full_submodule_name.split('.')
                    if any(
                        part.startswith('adaln_modulation') or part == 'cross_attn'
                        for part in parts
                    ):
                        continue
                    target_linear_modules.add(full_submodule_name)
        target_linear_modules = list(target_linear_modules)

        if is_main_process():
            print(f'[IC-LoRA V2] LoRA targets: {len(target_linear_modules)} linear modules (excluding adaln_modulation, cross_attn)')

        adapter_type = adapter_config['type']
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=target_linear_modules
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        self.peft_config = peft_config
        self.lora_model = peft.get_peft_model(self.transformer, peft_config)
        if is_main_process():
            self.lora_model.print_trainable_parameters()
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])
