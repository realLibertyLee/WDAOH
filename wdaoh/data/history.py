"""Causal EPA windows with station and supplier axes bound by the training-period geometry."""

import numpy as np
import pandas as pd

from wdaoh.data.context import graph


def causal(values, stations, origins, hours):
    """Return newest-first native values ending at each origin. Missing stations and pre-2021 hours yield NaN."""
    times = origins[..., None] - np.arange(hours)
    if np.any(origins < 0) or np.any(origins >= values.shape[1]):
        raise ValueError("Origin outside the native hourly matrix")
    result = values[np.maximum(0, stations[..., None]), np.maximum(0, times)].copy()
    result[(times < 0) | (stations[..., None] < 0)] = np.nan
    return result


def window_reader(bank, pol, roster, auxiliaries):
    """Bind five neighbors, inverse-distance weights and colocated species to the training-selected station axis."""
    values, stations = bank[pol]
    topology = graph(stations, roster)
    neighbors = topology["nodes"][:, 1:]
    distances = topology["distance_km"][roster]
    weights = np.zeros((len(stations), 5), dtype="float32")
    inverse = 1 / (distances + 1e-6)
    weights[roster] = (inverse / inverse.sum(axis=1, keepdims=True)).astype("float32")
    geo = stations[["lat", "lon"]].to_numpy()
    lookup = {}
    for name in auxiliaries:
        _, other = bank[name]
        idx = pd.Index(other.site).get_indexer(stations.site)
        good = idx >= 0
        good[good] &= (
            np.max(abs(other[["lat", "lon"]].to_numpy()[idx[good]] - geo[good]), axis=1)
            <= 0.001
        )
        lookup[name] = np.where(good, idx, -1)

    def windows(fields):
        site, tick = fields["site_index"], fields["origin_index"]
        nodes = neighbors[site]
        return dict(
            target=causal(values, site, tick, 169),
            neighbors=causal(
                values, nodes, np.broadcast_to(tick[:, None], nodes.shape), 25
            ),
            weights=weights[site],
            aux={
                k: causal(bank[k][0], indices[site], tick, 25)
                for k, indices in lookup.items()
            },
        )

    return windows


def feature_long(net, geo, s, t, length=168):
    """Build the native 168-hour target sequence using the same channels as its final 24 hours."""
    hour = t[:, None] + np.arange(-length + 1, 1)[None, :]
    assert np.all(hour <= t[:, None])
    valid = hour >= 0
    v = np.asarray(net[s[:, None], np.maximum(hour, 0)], dtype="float32").copy()
    v[~valid] = np.nan
    dates = pd.to_datetime(
        pd.Timestamp("2021-01-01", tz="UTC").value + hour.ravel() * 3600 * 10**9,
        utc=True,
    )
    parts = [
        dates.hour.to_numpy() / 24,
        dates.dayofweek.to_numpy() / 7,
        (dates.dayofyear.to_numpy() - 1) / 365.25,
    ]
    cols = []
    for part in parts:
        cols.extend(
            [
                np.sin(2 * np.pi * part).reshape(-1, length),
                np.cos(2 * np.pi * part).reshape(-1, length),
            ]
        )
    cols.extend(
        [
            np.broadcast_to(geo[s, 0, None] / 90, v.shape),
            np.broadcast_to(geo[s, 1, None] / 180, v.shape),
            np.isfinite(v).astype("float32"),
            v,
        ]
    )
    return np.stack(cols, -1).astype("float32")
