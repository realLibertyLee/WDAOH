"""Paper comparisons across tasks, using new reader forecasts and a common physical-station resample."""

import json
from collections import defaultdict
from dataclasses import replace
from itertools import product

import numpy as np

from wdaoh.evaluation.comparisons import (
    POPULATIONS,
    REPLICATES,
    interval,
    load_rows,
    site_contrast,
    station_totals,
    task_intervals,
)
from wdaoh.evaluation.metrics import paired
from wdaoh.experiments.design import REFERENCE
from wdaoh.protocol import HORIZONS, POLLUTANTS, SEEDS, digest, identity, settings


def decision(task):
    """Read the validation choice named by the held-out forecast and verify its binding."""
    path = (
        task.work
        / "decisions"
        / task.case
        / task.model
        / task.experiment
        / "freeze.json"
    )
    run = json.loads((task.output / "forecast.json").read_text())
    if run["freeze_sha256"] != digest(path):
        raise ValueError("Forecast does not match validation freeze")
    return json.loads(path.read_text()), run


def records(task, choice, run, arm, group, year):
    """Load verified forecast rows on the physical-station axis, for either evaluation year."""
    if year == 2025:
        return load_rows(task.output, arm, group, run)
    directory = task.work / choice["fits"][arm]
    path = directory / ("validation_" + group + ".npz")
    ledger = directory / "validation.json"
    if (
        digest(ledger) != choice["validation_sha256"][arm]
        or digest(path) != json.loads(ledger.read_text())["artifacts"][path.name]
    ):
        raise ValueError("Changed 2024 predictions")
    with np.load(path, allow_pickle=False) as z:
        row = dict(z)
    site_ids = json.loads((task.inputs / REFERENCE / "samples.json").read_text())[
        "station_ids"
    ]
    row["site"] = np.asarray(site_ids)[row["site_index"]]
    return row


def same_targets(rows):
    """Reject a reordered target, changed physical station or different observed label across predictors."""
    for row in rows[1:]:
        paired(rows[0], row)
        if not np.array_equal(rows[0]["site"], row["site"]):
            raise ValueError("Forecasts use different physical station identifiers")


def factorial_panel(tasks, experiment):
    """Aggregate signed normalized-MAE contrasts across 18 tasks and five seeds on one global station frame."""
    if {(t.pollutant, t.horizon, t.seed) for t in tasks} != set(
        product(POLLUTANTS, HORIZONS, SEEDS)
    ) or len(tasks) != 90:
        raise ValueError("Source panels require the complete 18 by 5 design")
    roster = sorted(
        {
            str(row[0])
            for net in settings("epa_networks").values()
            for row in net["network"]
        }
    )
    station = np.asarray(roster)
    weights = np.random.default_rng(20261102).multinomial(
        len(roster), np.full(len(roster), 1 / len(roster)), size=REPLICATES
    )
    contrasts = settings("contrasts")[experiment]
    arms = list(dict.fromkeys(k for terms in contrasts.values() for k in terms))
    results = []
    for year in (2024, 2025):
        subtotal = {
            g: np.zeros((1 + REPLICATES, len(arms)), dtype=float) for g in POPULATIONS
        }
        counts = dict.fromkeys(POPULATIONS, 0)
        for task in tasks:
            freeze, run = decision(task)
            for group in (g for g in POPULATIONS if g != "LMH"):
                rows = [records(task, freeze, run, a, group, year) for a in arms]
                same_targets(rows)
                sites, index = np.unique(rows[0]["site"], return_inverse=True)
                idx = np.searchsorted(station, sites)
                if not np.array_equal(station[idx], sites):
                    raise ValueError("Unregistered physical station")
                n = np.bincount(index)
                errs = np.stack(
                    [
                        np.abs(r["prediction"].astype(float) - r["y"].astype(float))
                        for r in rows
                    ],
                    axis=1,
                )
                if not len(errs):
                    raise ValueError("Empty source-panel support group")
                subtotal[group][0] += errs.mean(0) / run["reference_std"]
                if experiment != "components":
                    sums = np.stack(
                        [np.bincount(index, weights=e) for e in errs.T], axis=1
                    )
                    w = weights[:, idx].astype(float)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        normalized = (
                            (w @ sums) / (w @ n)[:, None] / run["reference_std"]
                        )
                    subtotal[group][1:] += normalized.astype(np.float32)
                counts[group] += len(errs)
        subtotal["LMH"] = sum(subtotal[k] for k in ("low", "medium", "high")) / 3
        counts["LMH"] = sum(counts[k] for k in ("low", "medium", "high"))
        for group, panel in subtotal.items():
            for name, terms in contrasts.items():
                gain = -(panel @ np.array([terms.get(a, 0) for a in arms])) / len(tasks)
                results.append(
                    dict(
                        year=year,
                        population=group,
                        contrast=name,
                        settings=len(tasks),
                        memberships=counts[group],
                        global_site_frame=len(roster),
                        positive_gain=float(gain[0]),
                        gain_ci=interval(gain[1:])
                        if experiment != "components"
                        else None,
                    )
                )
    return results


def selected_contrast(tasks, other_model=None):
    """Compare the LMH-selected predictor with R1 target-only or with a second independently selected EEA model."""
    values = {g: {} for g in POPULATIONS}
    context_choices = 0
    for task in tasks:
        choice, run = decision(task)
        arm = choice["choices"]["sparse_equal3"]
        context_choices += int(arm == "S1_CONTEXT")
        comparator = (
            replace(task, model=other_model)
            if other_model
            else replace(task, experiment="joint")
        )
        alternative, other = decision(comparator)
        other_arm = (
            alternative["choices"]["sparse_equal3"] if other_model else REFERENCE
        )
        if other["reference_std"] != run["reference_std"]:
            raise ValueError("Training label scale differs for paired contrasts")
        key = (task.pollutant, task.horizon, task.seed)
        for g in (g for g in POPULATIONS if g != "LMH"):
            first = records(task, choice, run, arm, g, 2025)
            second = records(comparator, alternative, other, other_arm, g, 2025)
            same_targets([first, second])
            target = first["y"].astype(float)
            delta = np.abs(second["prediction"].astype(float) - target) - np.abs(
                first["prediction"].astype(float) - target
            )
            values[g][key] = float(delta.mean() / run["reference_std"])
        values["LMH"][key] = float(
            np.mean([values[g][key] for g in ("low", "medium", "high")])
        )
    return [
        dict(
            population=g,
            first_model=tasks[0].model,
            comparator=other_model or "target-only",
            positive_gain="comparator minus first",
            joint_selected=context_choices,
            settings=len(tasks),
            **task_intervals(
                [[v[p, h, s] for s in SEEDS] for p, h in product(POLLUTANTS, HORIZONS)]
            ),
        )
        for g, v in values.items()
    ]


def fixed_validation_panel(tasks):
    """Use each task’s original 5k-label SD for all fixed temporal or budget contrasts in 2024."""
    scores = defaultdict(dict)
    contrasts = settings("contrasts")[tasks[0].experiment]
    for task in tasks:
        frozen, run = decision(task)
        key = (task.pollutant, task.horizon, task.seed)
        values = {}
        for group in (g for g in POPULATIONS if g != "LMH"):
            rows = [records(task, frozen, run, arm, group, 2024) for arm in task.arms()]
            same_targets(rows)
            values[group] = {
                arm: float(
                    np.abs(r["prediction"].astype(float) - r["y"].astype(float)).mean()
                    / run["reference_std"]
                )
                for arm, r in zip(task.arms(), rows)
            }
        values["LMH"] = {
            arm: float(np.mean([values[g][arm] for g in ("low", "medium", "high")]))
            for arm in task.arms()
        }
        for group, arms in values.items():
            for label, terms in contrasts.items():
                scores[group, label][key] = sum(
                    arms[a] * coef for a, coef in terms.items()
                )
    return [
        dict(
            year=2024,
            population=g,
            contrast=label,
            **task_intervals(
                [[v[p, h, s] for s in SEEDS] for p, h in product(POLLUTANTS, HORIZONS)]
            ),
        )
        for (g, label), v in scores.items()
    ]


def graph_panels(tasks):
    """Compute both registered graph years, with paired site resampling across five seeds within each setting."""
    blocks = defaultdict(list)
    for task in tasks:
        blocks[task.pollutant, task.horizon].append(task)
    result = []
    for (pol, horizon), block in blocks.items():
        if sorted(t.seed for t in block) != list(SEEDS):
            raise ValueError("Graph contrasts need each registered seed once")
        fits = [(task, *decision(task)) for task in block]
        for year in (2024, 2025):
            for name, terms in settings("contrasts")["graph"].items():
                for population, groups in POPULATIONS.items():
                    parts = []
                    for task, frozen, run in fits:
                        for g in groups:
                            rows = [
                                records(task, frozen, run, arm, g, year)
                                for arm in terms
                            ]
                            parts.append(station_totals(rows, run["reference_std"]))
                    stats = site_contrast(
                        parts,
                        list(terms.values()),
                        np.random.default_rng(20261102),
                        physical=True,
                    )
                    value = stats["metrics"]["nMAE"]
                    ci = value["ci"]
                    result.append(
                        dict(
                            pollutant=pol,
                            horizon=horizon,
                            year=year,
                            contrast=name,
                            population=population,
                            positive_gain=-value["value"]
                            if value["value"] is not None
                            else None,
                            gain_ci_low=-ci["upper"] if ci["valid"] >= 1900 else None,
                            gain_ci_high=-ci["lower"] if ci["valid"] >= 1900 else None,
                            **stats,
                        )
                    )
    return result


def winner_frequencies(tasks):
    """Count validation winners and estimate performance conditional on a winning view with paired 18-task draws."""
    ordered = list(product(POLLUTANTS, HORIZONS))
    if len(tasks) != len(ordered) * len(SEEDS) or {
        (t.pollutant, t.horizon, t.seed) for t in tasks
    } != set(product(POLLUTANTS, HORIZONS, SEEDS)):
        raise ValueError("View frequencies require a complete model and collection")
    seed = 20261104 + int(identity(["task_cluster_primary"])[:8], 16)
    weights = np.random.default_rng(seed).multinomial(
        18, np.full(18, 1 / 18), size=REPLICATES
    )
    bins = defaultdict(list)
    for t in tasks:
        freeze, run = decision(t)
        metrics = json.loads((t.output / "metrics.json").read_text())["arms"]
        for criterion, arm in freeze["choices"].items():
            for pop in POPULATIONS:
                record = metrics[arm]
                score = (record["LMH"] if pop == "LMH" else record["groups"][pop])[
                    "nMAE"
                ]
                bins[criterion, arm, pop].append(
                    (ordered.index((t.pollutant, t.horizon)), score)
                )
    result = []
    for criterion in sorted({key[0] for key in bins}):
        for arm in tasks[0].arms():
            for pop in POPULATIONS:
                rr = bins[criterion, arm, pop]
                idx = [i for i, _ in rr]
                scores = [v for _, v in rr]
                count = np.bincount(np.array(idx, dtype=int), minlength=len(ordered))
                sums = np.bincount(
                    np.array(idx, dtype=int), weights=scores, minlength=len(ordered)
                )
                den = weights @ count
                draws = np.divide(
                    weights @ sums, den, out=np.full(REPLICATES, np.nan), where=den > 0
                )
                result.append(
                    dict(
                        criterion=criterion,
                        arm=arm,
                        population=pop,
                        wins=len(rr),
                        settings=len(tasks),
                        held_out_nMAE=float(np.mean(scores)) if rr else None,
                        conditional_interval=interval(draws),
                    )
                )
    return result
