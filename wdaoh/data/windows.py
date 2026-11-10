"""UTC completion-hour grid, causal history channels and frozen sample populations."""

import numpy as np
import pandas as pd

GROUPS = {
    "all_strata": (0, 24),
    "zero": (0, 0),
    "near_no_history": (1, 4),
    "low": (5, 11),
    "medium": (12, 17),
    "high": (18, 24),
}
QUOTAS = {"low": 1667, "medium": 1667, "high": 1666}
HOUR = 3_600_000_000_000
START = pd.Timestamp("2021-01-01", tz="UTC").value
GRID = pd.date_range("2021-01-01", "2026-01-01", freq="h", inclusive="left", tz="UTC")
YEARS = GRID.year.to_numpy()
LENGTH = len(GRID)


def history_count(values):
    """Count finite native observations in each complete 24-hour window ending at the forecast origin."""
    finite = np.isfinite(values)
    cumulative = np.pad(np.cumsum(finite, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
    counts = np.zeros(finite.shape, dtype=np.int16)
    counts[:, 23:] = cumulative[:, 24:] - cumulative[:, :-24]
    return counts


def history_matrix(values, geo, site, origin, channels=10):
    """Construct target-last historical inputs with calendar, coordinates and observed-value masks."""
    hours = origin[:, None] + np.arange(-23, 1)
    date = pd.to_datetime(START + hours.ravel() * HOUR, utc=True)
    phases = (
        date.hour.to_numpy() / 24,
        date.dayofweek.to_numpy() / 7,
        (date.dayofyear.to_numpy() - 1) / 365.25,
    )
    data = np.full((len(site), 24, channels), np.nan, dtype=np.float32)
    clock = np.stack(
        [f(2 * np.pi * angle) for angle in phases for f in (np.sin, np.cos)], axis=-1
    )
    data[:, :, :6] = clock.reshape(len(site), 24, 6)
    data[:, :, 6] = geo[site, 0, None] / 90
    data[:, :, 7] = geo[site, 1, None] / 180
    history = values[site[:, None], hours]
    data[:, :, 8] = np.isfinite(history)
    data[:, :, -1] = history
    return data


def eea_groups(values, counts, natural, seed, pollutant_code, horizon):
    """Sample station/hour keys in frozen RNG order; omit history windows affected by source exclusions."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, pollutant_code, horizon]))
    picked = {}
    for stage, years, groups in (
        ("train", [2021, 2022, 2023], QUOTAS),
        ("validation", [2024], GROUPS),
        ("test", [2025], GROUPS),
    ):
        random = (
            np.random.default_rng(
                np.random.SeedSequence([seed, pollutant_code, horizon, 2025])
            )
            if stage == "test"
            else rng
        )
        pieces = []
        for group in groups:
            low, high = GROUPS[group]
            origin = np.arange(23, LENGTH - horizon)
            valid = (
                np.isin(YEARS[origin], years)
                & np.isin(YEARS[origin + horizon], years)
                & np.isfinite(values[:, origin + horizon])
                & (counts[:, origin] == natural[:, origin])
                & (counts[:, origin] >= low)
                & (counts[:, origin] <= high)
            )
            site, relative = np.nonzero(valid)
            budget = QUOTAS[group] if stage == "train" else 600
            draw = random.choice(len(site), min(budget, len(site)), replace=False)
            chosen = np.column_stack((site[draw], origin[relative[draw]]))
            chosen = chosen[np.lexsort((chosen[:, 0], chosen[:, 1]))]
            if stage == "train":
                pieces.append(chosen)
            else:
                picked[stage + "_" + group] = chosen
        if pieces:
            picked["train"] = np.concatenate(pieces)
    return picked


def eea_payloads(
    values, geo, aux, neighbors, counts, natural, picked, horizon, with_persistence
):
    """Build matched 24-hour arms; fit channel means/deviations on training rows and preserve query masks."""
    for view in ("S0_TARGET24", "S1_CONTEXT"):
        frozen = {}

        def matrix(site, origin):
            x = history_matrix(
                values, geo, site, origin, 10 if view == "S0_TARGET24" else 30
            )
            if view == "S1_CONTEXT":
                hours = origin[:, None] + np.arange(-23, 1)
                for rank in range(5):
                    v = aux[site[:, None], rank, hours]
                    x[:, :, 9 + rank * 2] = v
                    x[:, :, 10 + rank * 2] = np.isfinite(v)
                    v = values[neighbors[site, rank, None], hours]
                    x[:, :, 19 + rank * 2] = v
                    x[:, :, 20 + rank * 2] = np.isfinite(v)
            return x

        site, origin = picked["train"].T
        train = matrix(site, origin)
        y = values[site, origin + horizon]
        n = np.maximum(np.isfinite(train).sum((0, 1)), 1)
        mean = np.nansum(train, (0, 1)) / n
        sd = np.sqrt(np.nansum((train - mean) ** 2, (0, 1)) / n)
        stats = dict(
            mean=mean.astype("float32"),
            std=np.where(sd < 1e-6, 1, sd).astype("float32"),
            y_mean=np.float32(y.mean()),
            y_std=np.float32(max(float(y.std()), 1e-6)),
        )
        frozen["normalizer"] = stats
        for group, selected in picked.items():
            site, origin = selected.T
            x = train if group == "train" else matrix(site, origin)
            flat = (
                np.nan_to_num(
                    (x - stats["mean"]) / stats["std"], nan=0, posinf=0, neginf=0
                )
                .astype("float32")
                .reshape(len(site), 24 * x.shape[2])
            )
            history = x[:, :, -1]
            finite = np.isfinite(history)
            last = 23 - np.argmax(finite[:, ::-1], axis=1)
            frozen[group] = dict(
                X=flat,
                y=values[site, origin + horizon],
                site_index=site,
                origin_index=origin,
                target_index=origin + horizon,
                history_observed_count=counts[site, origin],
                natural_history_observed_count=natural[site, origin],
            )
            if with_persistence:
                frozen[group]["persistence"] = np.where(
                    finite.any(1), history[np.arange(len(site)), last], np.nan
                )
        yield view, frozen


def epa_payloads(values, geo, roster, seed, horizon):
    """Rebuild the primary target-history tensors in EPA native units using frozen support and seed offsets."""
    observed, counts = np.isfinite(values[roster]), history_count(values[roster])

    def choose(group, begin, stop, budget, seed, axis_end):
        low, high = GROUPS[group]
        origins = np.arange(axis_end - horizon)
        yes = (
            observed[:, horizon:axis_end]
            & (origins >= begin)
            & (origins + horizon < stop)
            & (counts[:, : axis_end - horizon] >= low)
            & (counts[:, : axis_end - horizon] <= high)
        )
        available = np.flatnonzero(yes)
        draw = np.random.default_rng(seed).choice(
            available, min(budget, len(available)), replace=False
        )
        index, origin = np.unravel_index(draw, yes.shape)
        site = roster[index]
        order = np.lexsort((site, origin + horizon))
        return np.column_stack((site[order], origin[order]))

    groups = {}
    parts = []
    for j, (group, n) in enumerate(QUOTAS.items()):
        parent = choose(group, 24, 26280, 5000, seed * 1000 + horizon, 35064)
        ids = np.random.default_rng(
            3000000 + seed * 1000 + horizon + j * 100000
        ).choice(5000, n, replace=False)
        parts.append(parent[ids])
    train = np.concatenate(parts)
    groups["train"] = train[np.lexsort((train[:, 0], train[:, 1] + horizon))]
    for i, group in enumerate(GROUPS):
        val_seed = seed * 1000 + horizon + 1 + (2000000 + i if i < 3 else 0)
        groups["validation_" + group] = choose(
            group, 26280, 35064, 600, val_seed, 35064
        )
        groups["test_" + group] = choose(
            group, 35064, LENGTH, 600, 4000000 + seed * 1000 + horizon + i, LENGTH
        )
    result = {}
    for name, selected in groups.items():
        site, origin = selected.T
        x = history_matrix(values, geo, site, origin)
        history = x[:, :, -1]
        finite = np.isfinite(history)
        last = np.where(finite, np.arange(24), -1).max(1)
        present = last >= 0
        persist = np.full(len(site), np.nan, np.float32)
        persist[present] = history[np.flatnonzero(present), last[present]]
        result[name] = dict(
            X=x,
            y=values[site, origin + horizon],
            site_index=site,
            origin_index=origin,
            target_index=origin + horizon,
            persistence=persist,
            persistence_lag_hours=np.where(present, 23 - last, -1),
            history_observed_count=finite.sum(1),
        )
    return result
