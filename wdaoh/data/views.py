"""Transform forecast inputs while preserving ordered labels, masks and time bounds."""

import numpy as np

from wdaoh.experiments.design import COMPONENTS, CONTEXT, FACTORIAL, REFERENCE
from wdaoh.protocol import settings


def input_arm(arm):
    """Map an information intervention to its underlying matched set of stored samples."""
    if arm in (REFERENCE, *COMPONENTS):
        return REFERENCE
    if arm in (
        *FACTORIAL,
        "P3_COMPACT11",
        "SELF_ONLY",
        "GRAPH_NEIGHBORS",
        "MASKED_NEIGHBOR_POOL",
    ):
        return CONTEXT
    return arm


def observation_arm(query, arm, pollutant):
    """Disable unused value/mask channels in the fixed-width EPA factorial and compact panels."""
    if arm not in (*FACTORIAL, *COMPONENTS, "P3_COMPACT11"):
        return query
    x = query["X"].copy()
    if arm in FACTORIAL:
        if arm[3] == "0":
            x[:, :, 9:19:2] = np.nan
            x[:, :, 10:19:2] = 0
        if arm[4] == "0":
            x[:, :, 19:29:2] = np.nan
            x[:, :, 20:29:2] = 0
    elif arm == "P3_COMPACT11":
        keep = settings("compact_auxiliaries")[pollutant]
        pollutants = ("CO", "NO2", "SO2", "PM25", "PM10", "O3")
        for i, k in enumerate(p for p in pollutants if p != pollutant):
            if k not in keep:
                x[:, :, 9 + 2 * i] = np.nan
                x[:, :, 10 + 2 * i] = 0
    else:
        if arm in ("P2_V", "P2_VC"):
            x[:, :, 8] = 0
        if arm in ("P2_V", "P2_VM"):
            x[:, :, :8] = 0
    return dict(query, X=x)
