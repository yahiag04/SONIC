"""Removal operators: subtract a separated target estimate from the mixture.

The baseline subtracts everywhere; SONIC subtracts only where the temporal
mask is non-zero, so samples with mask == 0 are returned unchanged.
"""

from __future__ import annotations

import numpy as np


def _check(x: np.ndarray, s_hat: np.ndarray) -> None:
    if x.shape != s_hat.shape:
        raise ValueError(f"mixture shape {x.shape} != estimate shape {s_hat.shape}")


def global_remove(x: np.ndarray, s_hat: np.ndarray) -> np.ndarray:
    """Baseline: y = x - s_hat over the whole recording."""
    _check(x, s_hat)
    return (x - s_hat).astype(np.float32)


def masked_remove(x: np.ndarray, s_hat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """SONIC: y = x - mask * s_hat, with mask in [0, 1].

    Where mask == 0 the input samples are copied verbatim (bit-identical).
    """
    _check(x, s_hat)
    if mask.shape != x.shape:
        raise ValueError(f"mask shape {mask.shape} != mixture shape {x.shape}")
    y = np.array(x, dtype=np.float32, copy=True)
    active = mask > 0
    y[active] = (x[active] - mask[active] * s_hat[active]).astype(np.float32)
    return y
