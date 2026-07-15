"""Pure helpers for the official Krea 2 flow-matching timestep schedule."""

from __future__ import annotations

import math


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
