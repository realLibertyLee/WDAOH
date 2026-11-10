"""Original per-channel training-only normalization."""

import numpy as np


def fit_stats(x, y):
    """Estimate per-channel moments over observed training sample-hours and the training-label SD."""
    finite = np.isfinite(x)
    count = finite.sum(axis=(0, 1))
    mean = np.nansum(x, axis=(0, 1)) / np.maximum(count, 1)
    var = np.nansum((x - mean) ** 2, axis=(0, 1)) / np.maximum(count, 1)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1, std)
    return (
        mean.astype("float32"),
        std.astype("float32"),
        float(np.mean(y)),
        max(float(np.std(y)), 1e-6),
    )


def norm(x, st):
    """Apply the training channel statistics and map unobserved values to normalized zero."""
    return np.nan_to_num((x - st[0]) / st[1], nan=0, posinf=0, neginf=0).astype(
        "float32"
    )
