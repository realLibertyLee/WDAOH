"""Lossless permutation of flat input fields into a target-last sequence for DLinear+side."""

import numpy as np


def indices(columns):
    """Map the 24 named target lags and all remaining side fields to disjoint column indices."""
    target = [columns.index("value_lag" + str(k)) for k in range(24)]
    side = [i for i in range(len(columns)) if i not in target]
    return target, side


def packed(x, target, side):
    """Put target lags last and preserve every side field in a zero-padded 24-step sequence."""
    channels = 1 + (len(side) + 23) // 24
    out = np.zeros((len(x), 24, channels), dtype="float32")
    out[:, :, -1] = x[:, target[::-1]]
    flat = np.zeros((len(x), 24 * (channels - 1)), dtype="float32")
    flat[:, : len(side)] = x[:, side]
    out[:, :, :-1] = flat.reshape(len(x), 24, channels - 1)
    return out
