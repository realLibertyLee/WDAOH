"""Fixed colocated and neighboring stations and nested EPA support budgets."""

import io

import numpy as np
import pandas as pd

from wdaoh.data import windows
from wdaoh.data.eea import CODES

KEYS = ("site_index", "origin_index", "target_index")


def context_mapper(bank, pol, roster):
    """Bind colocated fixed POCs within 0.001° and five distance-ranked neighbors in this seed's target pool."""
    values, sites = bank[pol]
    geo = sites[["lat", "lon"]].to_numpy()
    c = np.radians(geo[roster])
    d = c[:, None, :] - c[None, :, :]
    dist = (
        2
        * 6371
        * np.arcsin(
            np.sqrt(
                np.clip(
                    np.sin(d[:, :, 0] / 2) ** 2
                    + np.cos(c[:, None, 0])
                    * np.cos(c[None, :, 0])
                    * np.sin(d[:, :, 1] / 2) ** 2,
                    0,
                    1,
                )
            )
        )
    )
    np.fill_diagonal(dist, np.inf)
    neighbors = np.zeros((len(sites), 5), int)
    neighbors[roster] = roster[np.argsort(dist, axis=1, kind="stable")[:, :5]]
    aux = []
    for other in CODES:
        if other == pol:
            continue
        obs, stations = bank[other]
        lookup = dict(zip(stations.site, stations.index))
        which = np.array([lookup.get(site, -1) for site in sites.site])
        good = which >= 0
        good[good] &= (
            np.max(
                abs(stations[["lat", "lon"]].to_numpy()[which[good]] - geo[good]),
                axis=1,
            )
            <= 0.001
        )
        aux.append((obs, np.where(good, which, -1)))

    def context(payload):
        si, t = payload["site_index"], payload["origin_index"]
        ticks = t[:, None] + np.arange(-23, 1)
        cols = []
        for obs, lookup in aux:
            ix = lookup[si]
            v = np.full((len(t), 24), np.nan, np.float32)
            ok = ix >= 0
            v[ok] = obs[ix[ok, None], ticks[ok]]
            cols.extend([v, np.isfinite(v).astype("float32")])
        for k in range(5):
            v = values[neighbors[si, k, None], ticks]
            cols.extend([v, np.isfinite(v).astype("float32")])
        x = payload["X"]
        return dict(
            payload,
            X=np.concatenate(
                [x[:, :, :-1], np.stack(cols, axis=-1), x[:, :, -1:]], axis=-1
            ),
        )

    return context


def nested_keys(values, roster, train, pol, seed, horizon):
    """Recreate the three 5k parent strata and nested support draws using only 2021–23 targets."""
    n = train["history_observed_count"]
    rng = np.random.default_rng(20261031 + horizon + list(CODES).index(pol) * 1000)
    small = np.concatenate(
        [
            rng.choice(np.flatnonzero((n >= lo) & (n <= hi)), qty, replace=False)
            for (lo, hi), qty in zip(((5, 11), (12, 17), (18, 24)), (334, 333, 333))
        ]
    )
    small = small[
        np.lexsort((train["site_index"][small], train["target_index"][small]))
    ]
    known = set(zip(train["site_index"], train["origin_index"]))
    observed = np.isfinite(values[roster, :35064])
    counts = windows.history_count(values[roster, :35064])
    origin = np.arange(35064 - horizon)
    time_ok = (origin >= 24) & (origin + horizon < 26280)
    extra = []
    for (lo, hi), needed in zip(((5, 11), (12, 17), (18, 24)), (1667, 1666, 1667)):
        ok = (
            observed[:, horizon:]
            & time_ok
            & (counts[:, :-horizon] >= lo)
            & (counts[:, :-horizon] <= hi)
        )
        draw = np.random.default_rng(seed * 1000 + horizon).choice(
            np.flatnonzero(ok), 5000, replace=False
        )
        ix, orig = np.unravel_index(draw, ok.shape)
        site = roster[ix]
        order = np.lexsort((site, orig + horizon))
        pool = [(site[i], orig[i]) for i in order if (site[i], orig[i]) not in known]
        extra.extend(pool[i] for i in rng.choice(len(pool), needed, replace=False))
    return small, np.asarray(extra).T


def graph(stations: pd.DataFrame, roster: np.ndarray) -> dict:
    """Construct each enrolled station's six-node graph with the historical CSV distance precision, in km."""
    geo = stations[["lat", "lon"]].to_numpy()
    coord = np.radians(geo[roster])
    delta = coord[:, None, :] - coord[None, :, :]
    angular = (
        np.sin(delta[:, :, 0] / 2) ** 2
        + np.cos(coord[:, None, 0])
        * np.cos(coord[None, :, 0])
        * np.sin(delta[:, :, 1] / 2) ** 2
    )
    distance = 2 * 6371 * np.arcsin(np.sqrt(np.clip(angular, 0, 1)))
    np.fill_diagonal(distance, np.inf)
    selected = np.argsort(distance, axis=1, kind="stable")[:, :5]
    km = np.take_along_axis(distance, selected, axis=1)
    # Historical training read back the textual distance column of the generated source topology.
    text = pd.DataFrame({"distance_km": km.ravel()}).to_csv(index=False)
    km = pd.read_csv(io.StringIO(text)).distance_km.to_numpy().reshape(len(roster), 5)
    nodes = np.full((len(stations), 6), -1, dtype=np.int64)
    distances = np.full((len(stations), 5), np.nan)
    adjacent = np.broadcast_to(
        np.eye(6, dtype=np.float32), (len(stations), 6, 6)
    ).copy()
    nodes[roster, 0] = roster
    nodes[roster, 1:] = roster[selected]
    distances[roster] = km
    weights = 1 / np.maximum(km, 1.0)
    adjacent[roster, 0, 1:] = weights / weights.sum(axis=1, keepdims=True)
    adjacent[roster, 0, :] /= 2
    return dict(
        nodes=nodes,
        distance_km=distances,
        adjacency=adjacent,
        sites=stations.site.to_numpy(dtype=str),
        geo=geo,
    )
