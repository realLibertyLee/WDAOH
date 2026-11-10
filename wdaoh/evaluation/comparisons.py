"""Compare matched forecasts under the study's task and physical-station resampling designs."""

import json
from collections import defaultdict
from itertools import product

import numpy as np

from wdaoh.evaluation.metrics import paired
from wdaoh.experiments.design import CONTEXT, REFERENCE
from wdaoh.protocol import (
    GROUPS,
    HORIZONS,
    LMH,
    POLLUTANTS,
    SEEDS,
    digest,
    identity,
    settings,
)

REPLICATES = 2000
POPULATIONS = {**{group: (group,) for group in GROUPS}, "LMH": LMH}
STAT_KEYS = ("nMAE", "nRMSE", "relative_MAE_percent")


def interval(draws):
    """Return the 2.5th and 97.5th percentiles and the actual finite bootstrap denominator."""
    valid = np.asarray(draws)[np.isfinite(draws)]
    return dict(
        valid=len(valid),
        total=len(draws),
        lower=float(np.quantile(valid, 0.025)) if len(valid) else None,
        upper=float(np.quantile(valid, 0.975)) if len(valid) else None,
    )


def task_intervals(matrix):
    """Average five seeds within each of 18 tasks. Resample task blocks with shared draws."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (18, 5) or not np.isfinite(matrix).all():
        raise ValueError(
            "Task inference needs 18 complete pollutant-horizon blocks of five finite seeds"
        )

    def draws(count, label):
        seed = 20261104 + int(identity([label])[:8], 16)
        return (
            np.random.default_rng(seed).multinomial(
                count, np.full(count, 1 / count), size=REPLICATES
            )
            / count
        )

    tasks = draws(18, "task_cluster_primary")
    seeds = draws(5, "seed_cluster_robustness")
    return dict(
        mean=float(matrix.mean()),
        task_ci=interval(tasks @ matrix.mean(1)),
        crossed_ci=interval(np.einsum("bt,ts,bs->b", tasks, matrix, seeds)),
        seed_sd=float(matrix.mean(0).std(ddof=1)),
    )


def load_rows(path, arm, group, record):
    """Load a checksum-bound forecast population, retaining labels and ordered physical station keys."""
    f = path / arm / (group + ".npz")
    if digest(f) != record["arms"][arm][group]:
        raise ValueError("Changed prediction series")
    with np.load(f, allow_pickle=False) as z:
        return dict(z)


def station_totals(rows, reference_std):
    """Summarize aligned arm errors and denominators on one common physical-station axis."""
    for row in rows[1:]:
        paired(rows[0], row)
    sites, index = np.unique(rows[0]["site"], return_inverse=True)
    counts = np.bincount(index)
    absolute = []
    squared = []
    for row in rows:
        error = row["prediction"].astype(float) - row["y"].astype(float)
        absolute.append(np.bincount(index, weights=np.abs(error)))
        squared.append(np.bincount(index, weights=np.square(error)))
    return sites, counts, np.asarray(absolute), np.asarray(squared), reference_std


def site_components(path, terms, groups):
    """Tabulate paired counts, absolute errors and squared errors by physical site for each population."""
    run = json.loads((path / "forecast.json").read_text())
    parts = []
    for g in groups:
        rows = [load_rows(path, arm, g, run) for arm in terms]
        parts.append(station_totals(rows, run["reference_std"]))
    return parts


def site_contrast(
    parts, coefficients, rng, pooled_mae=False, insertion_order=False, physical=False
):
    """Use the same physical-site draws across arms, seeds and groups. Report undefined replicates explicitly."""
    frame = np.concatenate([p[0] for p in parts])
    sites = (
        np.array(list(dict.fromkeys(frame))) if insertion_order else np.unique(frame)
    )
    index = {s: i for i, s in enumerate(sites)}
    if len(sites) < 2:
        raise ValueError("Too few physical station clusters")
    c = np.array(coefficients, float)
    weights = rng.multinomial(
        len(sites), np.full(len(sites), 1 / len(sites)), size=REPLICATES
    ).astype(float)
    weights = np.vstack((np.ones(len(sites)), weights))
    agg = np.zeros((len(weights), 5))
    pooled = np.zeros((len(weights), 2))
    has_relative = len(c) == 2 and np.array_equal(c, [1.0, -1.0])
    for names, counts, ab, sq, sd in parts:
        w = weights[:, [index[name] for name in names]]
        den = w @ counts
        n = counts.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            mae = (w @ ab.T) / den[:, None]
            rmse = np.sqrt((w @ sq.T) / den[:, None])
            agg[:, 3] += mae @ c
            agg[:, 4] += rmse @ c
            agg[:, 0] += (mae @ c) / sd
            agg[:, 1] += (rmse @ c) / sd
            if has_relative:
                agg[:, 2] += np.divide(
                    100 * (mae @ c),
                    mae[:, 0],
                    out=np.full(len(den), np.nan),
                    where=mae[:, 0] > 0,
                )
            pooled[:, 0] += (w @ (c @ ab)) / (sd * n)
            pooled[:, 1] += den / n
    agg /= len(parts)
    if pooled_mae:
        agg[:, 0] = pooled[:, 0] / pooled[:, 1]
    cols = list(enumerate(STAT_KEYS if has_relative else STAT_KEYS[:2]))
    if physical:
        cols.extend([(3, "MAE"), (4, "RMSE")])
    return dict(
        stations=len(sites),
        rows=sum(int(p[1].sum()) for p in parts),
        metrics={
            name: dict(
                value=float(agg[0, i]) if np.isfinite(agg[0, i]) else None,
                ci=interval(agg[1:, i]),
            )
            for i, name in cols
        },
    )


def factorial_tables(tasks, experiment, model):
    """Compute all signed source, component or compact contrasts within each pollutant and horizon."""
    specs = settings("contrasts")[experiment]
    grouped = defaultdict(list)
    for task in tasks:
        grouped[(task.pollutant, task.horizon)].append(task)
    result = []
    for (pol, h), block in grouped.items():
        if {t.seed for t in block} != set(SEEDS):
            raise ValueError("Incomplete five-seed panel")
        for label, terms in specs.items():
            for pop, groups in POPULATIONS.items():
                parts = [
                    part
                    for task in block
                    for part in site_components(task.output, terms, groups)
                ]
                stats = site_contrast(
                    parts,
                    list(terms.values()),
                    np.random.default_rng(
                        20261102 if experiment == "graph" else 20261101
                    ),
                    physical=True,
                )
                result.append(
                    dict(
                        model=model,
                        pollutant=pol,
                        horizon=h,
                        contrast=label,
                        population=pop,
                        **stats,
                    )
                )
    return result


def selected_tables(tasks):
    """Summarize retained arms/selection criteria or paired fixed-arm differences on common evaluation keys."""
    series = defaultdict(dict)
    for task in tasks:
        metrics = json.loads((task.output / "metrics.json").read_text())
        choice = json.loads(
            (
                task.work
                / "decisions"
                / task.case
                / task.model
                / task.experiment
                / "freeze.json"
            ).read_text()
        )["choices"]
        arms = metrics["arms"]
        for pop in POPULATIONS:

            def score(arm):
                rec = arms[arm]
                return (rec["LMH"] if pop == "LMH" else rec["groups"][pop])["nMAE"]

            values = (
                {k: score(arm) for k, arm in choice.items()}
                if choice
                else {a: score(a) for a in arms}
            )
            if "all_strata" in choice:
                values["natural_choice_minus_LMH_choice"] = (
                    values["all_strata"] - values["sparse_equal3"]
                )
            elif not choice:
                for label, terms in (
                    settings("contrasts").get(task.experiment, {}).items()
                ):
                    values[label] = sum(
                        coef * score(arm) for arm, coef in terms.items()
                    )
            for name, value in values.items():
                series[name, pop][task.pollutant, task.horizon, task.seed] = value
    return [
        dict(
            estimand=name,
            population=pop,
            **task_intervals(
                [
                    [points[p, h, s] for s in SEEDS]
                    for p, h in product(POLLUTANTS, HORIZONS)
                ]
            ),
        )
        for (name, pop), points in series.items()
    ]


def r1_tables(tasks_by_model):
    """Reproduce the sequential shared-site RNG stream for all seven models, including the persistence offset."""
    order = (
        "CatBoost",
        "DLinear+side",
        "ModernTCN",
        "GRU-D",
        "iTransformer",
        "TabPFN",
        "TabICL",
    )
    if set(tasks_by_model) != set(order):
        raise ValueError(
            "R1 site intervals require all seven models and the complete frozen RNG stream"
        )
    rng = np.random.default_rng(20261031)
    for g in ("all_strata", "near_no_history", "low", "medium", "high"):
        frame = set()
        for task in tasks_by_model[order[0]]:
            q = task.sample(REFERENCE, "test_" + g)
            station_ids = json.loads(
                (task.inputs / REFERENCE / "samples.json").read_text()
            )["station_ids"]
            mask = np.isfinite(q["persistence"])
            frame.update(np.array(station_ids)[q["site_index"][mask]])
        k = len(frame)
        rng.multinomial(k, np.full(k, 1 / k), size=REPLICATES)
    result = []
    for model in order:
        tasks = sorted(
            tasks_by_model[model], key=lambda t: (t.pollutant, t.horizon, t.seed)
        )
        if len(tasks) != 90:
            raise ValueError("Expected 90 settings per model")
        for pop, groups in POPULATIONS.items():
            # The balanced R1 resample binds physical sites in first-seen order within low, medium and high.
            parts = [
                part
                for group in groups
                for t in tasks
                for part in site_components(t.output, (REFERENCE, CONTEXT), (group,))
            ]
            result.append(
                dict(
                    model=model,
                    population=pop,
                    **site_contrast(
                        parts,
                        [1, -1],
                        np.random.default_rng(20261102) if pop == "LMH" else rng,
                        pooled_mae=True,
                        insertion_order=pop == "LMH",
                    ),
                )
            )
    return result
