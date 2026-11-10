"""Support-fitted engineered features from causal target, neighbor and auxiliary windows."""

import numpy as np
import pandas as pd

from wdaoh.data.feature_blocks import atomic_blocks
from wdaoh.protocol import settings

ORIGIN = pd.Timestamp("2021-01-01", tz="UTC")
CFG = settings("feature_options")


def calendar(hours, kind="KDD"):
    """Encode UTC hour, weekday and annual phase on the registered calendar convention."""
    d = pd.to_datetime(
        ORIGIN.value + np.asarray(hours).ravel() * 3600 * 10**9, utc=True
    )
    phase = [
        d.hour.to_numpy() / 24,
        d.dayofweek.to_numpy() / 7,
        (
            (d.month.to_numpy() - 1) / 12
            if kind == "KDD"
            else (d.dayofyear.to_numpy() - 1) / 365.25
        ),
    ]
    return (
        np.stack([f(2 * np.pi * v) for v in phase for f in [np.sin, np.cos]], axis=-1)
        .astype("float32")
        .reshape(*np.shape(hours), 6)
    )


def stats(a, axis=-1):
    """Reduce over finite measurements, separately retaining the actual observation counts."""
    mask = np.isfinite(a)
    n = mask.sum(axis)
    den = np.maximum(n, 1)
    v = np.where(mask, a, 0.0).astype("float64")
    mean = v.sum(axis) / den
    sd = np.sqrt(np.maximum(0, (v * v).sum(axis) / den - mean * mean))
    lo = np.where(n, np.where(mask, a, np.inf).min(axis), 0)
    hi = np.where(n, np.where(mask, a, -np.inf).max(axis), 0)
    return dict(mean=mean, std=sd, min=lo, max=hi, count=n)


def normalize(a, st):
    """Apply support-fitted scaling while retaining unavailable observations."""
    return np.where(
        np.isfinite(a), (a - st["mean"]) / (st["std"] + 1e-5), np.nan
    ).astype("float32")


def train_scaler(values):
    """Fit location and spread on finite support-window occurrences, including overlapping windows."""
    v = np.concatenate([x.ravel() for x in values])
    v = v[np.isfinite(v)].astype("float64")
    return dict(
        mean=float(v.mean()) if len(v) else 0.0,
        std=float(v.std()) if len(v) else 1.0,
        finite_support_observation_occurrences=len(v),
        overlap_weighting="each permitted support-window occurrence; no validation fit",
        epsilon=1e-5,
    )


def gap(a):
    """Count elapsed hours since the most recent finite observation in a newest-first window."""
    m = np.isfinite(a)
    return np.where(m.any(-1), m.argmax(-1), a.shape[-1])


def features(w, origins, h, geo, station_missing, poll_missing, scalers):
    """Assemble one feature row per origin from causal measurements, geography and training-fitted statistics."""
    t = normalize(w["target"], scalers["target"])
    nb = normalize(w["neighbors"], scalers["target"])
    n = len(t)
    cols = {}
    for lag in range(24):
        cols[f"value_lag{lag}"] = np.nan_to_num(t[:, lag], nan=0)
        cols[f"mask_lag{lag}"] = np.isfinite(t[:, lag]).astype(float)
    names = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    for j, k in enumerate(names):
        cols[k] = calendar(origins)[:, j]
        cols["tgt_" + k] = calendar(origins + h)[:, j]
    cols["lat"] = geo[:, 0]
    cols["lon"] = geo[:, 1]
    count = np.isfinite(t[:, :24]).sum(1)
    available168 = np.minimum(origins + 1, 168)
    cols.update(
        valid_count_24h=count,
        missing_rate_24h=1 - count / 24,
        last_valid_gap=gap(t[:, :24]),
        consecutive_missing_len=gap(t[:, :24]),
        valid_count_7d=np.isfinite(t[:, :168]).sum(1),
        missing_rate_7d=1 - np.isfinite(t[:, :168]).sum(1) / available168,
        station_missing_rate_train=station_missing,
        pollutant_missing_rate_train=np.full(n, poll_missing),
    )
    for lag in CFG["extra_lags"]:
        cols[f"teng_value_lag_{lag}"] = np.nan_to_num(t[:, lag], nan=0)
        cols[f"teng_mask_lag_{lag}"] = np.isfinite(t[:, lag]).astype(float)
    for lag in CFG["temporal_trend_lags"]:
        valid = np.isfinite(t[:, 0]) & np.isfinite(t[:, lag])
        cols[f"teng_value_trend_{lag}h"] = np.where(valid, t[:, 0] - t[:, lag], 0)
        cols[f"teng_value_trend_valid_{lag}h"] = valid.astype(float)
    for window in CFG["temporal_windows"]:
        s = stats(t[:, :window])
        den = np.minimum(origins + 1, window)
        for k in ["mean", "std", "min", "max"]:
            cols[f"teng_value_{k}_{window}h"] = s[k]
        cols[f"teng_valid_count_{window}h"] = s["count"]
        cols[f"teng_missing_rate_{window}h"] = 1 - s["count"] / den
        cols[f"teng_last_valid_gap_{window}h"] = np.minimum(gap(t[:, :window]), den)
    weights = w["weights"]
    for typ, values in [
        ("lag", CFG["spatial_lags"]),
        ("window", CFG["spatial_windows"]),
    ]:
        for z in values:
            a = nb[:, :, z : z + 1] if typ == "lag" else nb[:, :, :z]
            mask = np.isfinite(a)
            v = np.where(mask, a, 0)
            support = mask.sum((1, 2))
            weighted = weights[:, :, None] * mask
            den = weighted.sum((1, 2))
            suffix = f"lag_{z}" if typ == "lag" else f"{z}h"
            cols[f"sp_k5_value_mean_{suffix}"] = v.sum((1, 2)) / np.maximum(support, 1)
            cols[f"sp_k5_value_idw_{suffix}"] = (v * weights[:, :, None]).sum(
                (1, 2)
            ) / np.where(den > 0, den, 1)
            width = 1 if typ == "lag" else np.minimum(origins + 1, z)
            cols[f"sp_k5_valid_rate_{suffix}"] = support / (5 * width)
    for pol, a in w["aux"].items():
        x = normalize(a, scalers[pol])
        prefix = "cross_" + pol.lower()
        for lag in CFG["cross_lags"]:
            cols[f"{prefix}_value_lag_{lag}"] = np.nan_to_num(x[:, lag], nan=0)
            cols[f"{prefix}_mask_lag_{lag}"] = np.isfinite(x[:, lag]).astype(float)
        for lag in CFG["cross_trend_lags"]:
            valid = np.isfinite(x[:, 0]) & np.isfinite(x[:, lag])
            cols[f"{prefix}_value_trend_{lag}h"] = np.where(
                valid, x[:, 0] - x[:, lag], 0
            )
            cols[f"{prefix}_value_trend_valid_{lag}h"] = valid.astype(float)
        for window in CFG["cross_windows"]:
            s = stats(x[:, :window])
            for k in ["mean", "std", "min", "max"]:
                cols[f"{prefix}_value_{k}_{window}h"] = s[k]
            cols[f"{prefix}_valid_count_{window}h"] = s["count"]
            cols[f"{prefix}_missing_rate_{window}h"] = 1 - s["count"] / np.minimum(
                origins + 1, window
            )
    frame = pd.DataFrame(cols, dtype="float32")
    blocks = atomic_blocks(frame)
    order = [k for v in blocks.values() for k in v]
    assert len(order) == len(set(order)) == len(frame.columns)
    return frame[order], blocks
