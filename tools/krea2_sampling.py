"""Pure helpers for the official Krea 2 flow-matching timestep schedule."""

from __future__ import annotations

import math


KREA2_INFERENCE_DEFAULTS = {
    # The runner uses standard CFG: uncond + scale * (cond - uncond).
    # Ostris exposes cond + scale * (cond - uncond), so Raw's official
    # guidance 4.5 is 5.5 in this runner.
    'raw': (28, 5.5),
    'turbo': (8, 1.0),
}


def resolve_krea2_inference_defaults(
    diffusion_model_path,
    *,
    variant: str = 'auto',
    steps: int | None = None,
    text_guidance: float | None = None,
) -> tuple[str, int, float]:
    """Resolve safe Krea 2 Raw/Turbo inference defaults.

    ``auto`` recognizes Turbo from the diffusion checkpoint path and otherwise
    falls back to Raw. Explicit CLI values always win, so experiments are not
    constrained by the recommended profiles.
    """
    if variant not in ('auto', 'raw', 'turbo'):
        raise ValueError("variant must be 'auto', 'raw', or 'turbo'")
    resolved_variant = variant
    if resolved_variant == 'auto':
        model_name = str(diffusion_model_path).lower()
        resolved_variant = 'turbo' if 'turbo' in model_name else 'raw'

    default_steps, default_guidance = KREA2_INFERENCE_DEFAULTS[resolved_variant]
    resolved_steps = default_steps if steps is None else int(steps)
    resolved_guidance = default_guidance if text_guidance is None else float(text_guidance)
    if resolved_steps <= 0:
        raise ValueError('steps must be positive')
    return resolved_variant, resolved_steps, resolved_guidance


def resolve_krea2_inference_mu(variant: str, mu: float | None = None) -> float | None:
    """Use Turbo's fixed shift while leaving Raw resolution-dependent."""
    if variant not in ('raw', 'turbo'):
        raise ValueError("variant must be 'raw' or 'turbo'")
    if mu is not None:
        return float(mu)
    return 1.15 if variant == 'turbo' else None


def build_krea2_timesteps(
    sequence_length: int,
    steps: int,
    *,
    min_resolution: int = 256,
    max_resolution: int = 1280,
    spatial_compression: int = 8,
    patch_size: int = 2,
    y1: float = 0.5,
    y2: float = 1.15,
    mu: float | None = None,
) -> tuple[list[float], float]:
    """Return Krea's shifted ``1 -> 0`` Euler schedule and the resolved mu.

    Krea's official sampler interpolates ``mu`` in image-token space and then
    applies the equivalent of a rational flow shift with multiplier
    ``exp(mu)``. Keeping this helper independent from Diffusers prevents the
    generic reference runner from silently falling back to another model's
    scheduler defaults.
    """
    if sequence_length <= 0:
        raise ValueError('sequence_length must be positive')
    if steps <= 0:
        raise ValueError('steps must be positive')
    if min_resolution <= 0 or max_resolution <= min_resolution:
        raise ValueError('max_resolution must be greater than min_resolution > 0')
    if spatial_compression <= 0 or patch_size <= 0:
        raise ValueError('spatial_compression and patch_size must be positive')

    token_stride = spatial_compression * patch_size
    x1 = (min_resolution // token_stride) ** 2
    x2 = (max_resolution // token_stride) ** 2
    if x2 <= x1:
        raise ValueError('resolution endpoints collapse to the same token count')

    resolved_mu = float(mu) if mu is not None else (
        (y2 - y1) / (x2 - x1) * sequence_length
        + (y1 - ((y2 - y1) / (x2 - x1)) * x1)
    )
    shift = math.exp(resolved_mu)

    timesteps = []
    for index in range(steps + 1):
        base = 1.0 - index / steps
        shifted = shift * base / (1.0 + (shift - 1.0) * base)
        timesteps.append(shifted)
    # Avoid tiny floating-point residue at the integration boundary.
    timesteps[0] = 1.0
    timesteps[-1] = 0.0
    return timesteps, resolved_mu
