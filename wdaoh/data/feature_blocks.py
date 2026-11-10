from __future__ import annotations

import re
from collections import OrderedDict
from typing import Iterable

import pandas as pd

BASE_TIME = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

BASE_STATIC = [
    "lat",
    "lon",
]

COMPACT_MISSINGNESS = [
    "valid_count_24h",
    "missing_rate_24h",
    "last_valid_gap",
    "consecutive_missing_len",
    "valid_count_7d",
    "missing_rate_7d",
    "station_missing_rate_train",
    "pollutant_missing_rate_train",
]


def natural_key(value: str):
    """Sort numbered lags numerically within a named block."""
    return [
        int(token) if token.isdigit() else token
        for token in re.split(r"(\d+)", str(value))
    ]


def ordered_unique(values: Iterable[str]) -> list[str]:
    """Deduplicate labels while preserving their input order."""
    result = []
    seen = set()

    for value in values:
        if value in seen:
            continue

        result.append(value)
        seen.add(value)

    return result


def matching(df: pd.DataFrame, predicate) -> list[str]:
    """Find and naturally order the columns satisfying a block predicate."""
    return sorted(
        [str(column) for column in df.columns if predicate(str(column))],
        key=natural_key,
    )


def atomic_blocks(
    df: pd.DataFrame,
) -> OrderedDict[str, list[str]]:
    """Assign feature columns to the registered temporal, spatial, auxiliary and availability blocks."""
    base = matching(
        df,
        lambda column: re.fullmatch(r"value_lag\d+", column) is not None,
    ) + [column for column in BASE_TIME + BASE_STATIC if column in df.columns]

    raw_masks = matching(
        df,
        lambda column: re.fullmatch(r"mask_lag\d+", column) is not None,
    )

    compact_missingness = [
        column for column in COMPACT_MISSINGNESS if column in df.columns
    ]

    temporal_value = matching(
        df,
        lambda column: (
            column.startswith("tgt_")
            or column.startswith("teng_value_lag_")
            or (column.startswith("teng_value_trend_") and "trend_valid" not in column)
            or re.fullmatch(
                r"teng_value_(mean|std|min|max)_\d+h",
                column,
            )
            is not None
        ),
    )

    temporal_availability = matching(
        df,
        lambda column: (
            column.startswith("teng_mask_lag_")
            or column.startswith("teng_valid_count_")
            or column.startswith("teng_missing_rate_")
            or column.startswith("teng_last_valid_gap_")
            or column.startswith("teng_value_trend_valid_")
        ),
    )

    spatial_value = matching(
        df,
        lambda column: column.startswith("sp_") and "_value_" in column,
    )

    spatial_availability = matching(
        df,
        lambda column: column.startswith("sp_") and "valid_rate" in column,
    )

    cross_value = matching(
        df,
        lambda column: (
            column.startswith("cross_")
            and "_value_" in column
            and "trend_valid" not in column
        ),
    )

    cross_availability = matching(
        df,
        lambda column: (
            column.startswith("cross_")
            and (
                "_mask_" in column
                or "_valid_count_" in column
                or "_missing_rate_" in column
                or "_trend_valid_" in column
            )
        ),
    )

    return OrderedDict(
        [
            ("B", ordered_unique(base)),
            ("R", raw_masks),
            ("M", compact_missingness),
            ("T", temporal_value),
            ("MT", temporal_availability),
            ("SV", spatial_value),
            ("SA", spatial_availability),
            ("CV", cross_value),
            ("CA", cross_availability),
        ]
    )
