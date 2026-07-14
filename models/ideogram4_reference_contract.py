"""Pure tensor helpers for the Ideogram 4 reference packing contract."""

import torch


REFERENCE_IMAGE_INDICATOR = 4
REFERENCE_CONTRACT_VERSION = 'ideogram4_reference_conditioning_v1'

# adaln_modulation input is a function of the timestep embedding only, so LoRA
# there cannot encode the ref->target relation; it can only re-tune global
# modulation. With reference rows pinned at a clean timestep the adaln pathway
# sees a shifted input distribution, making it the highest-risk place to adapt.
# kohya musubi-tuner's Ideogram4 LoRA also targets only attention/feed_forward.
LORA_FORBIDDEN_MODULE_PATTERNS = ('adaln_modulation',)


def split_lora_target_modules(module_names, forbidden_patterns=LORA_FORBIDDEN_MODULE_PATTERNS):
    """Split module names into (allowed, excluded) by forbidden substring."""
    allowed = []
    excluded = []
    for name in module_names:
        if any(pattern in name for pattern in forbidden_patterns):
            excluded.append(name)
        else:
            allowed.append(name)
    return allowed, excluded


def apply_reference_dropout(reference_latents, dropout_probability, *, enabled=True):
    """Zero whole reference samples while preserving a fixed sequence shape."""
    if not enabled or dropout_probability == 0:
        return reference_latents
    keep = (
        torch.rand(
            reference_latents.shape[0],
            device=reference_latents.device,
        )
        >= dropout_probability
    )
    keep = keep.view(-1, *([1] * (reference_latents.ndim - 1)))
    return reference_latents * keep.to(reference_latents.dtype)


def offset_reference_positions(image_positions, temporal_offset):
    """Copy image MRoPE positions and offset only their temporal coordinate."""
    reference_positions = image_positions.clone()
    reference_positions[:, 0] += temporal_offset
    return reference_positions


def build_model_timesteps(
    target_timesteps,
    sequence_length,
    reference_start,
    reference_model_timestep,
):
    """Convert flow timesteps and mark the reference span as permanently clean."""
    model_timesteps = (
        (1.0 - target_timesteps)
        .unsqueeze(1)
        .expand(-1, sequence_length)
        .clone()
    )
    model_timesteps[:, reference_start:] = reference_model_timestep
    return model_timesteps
