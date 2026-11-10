"""Fit and validate eligible arms, persist the decision, then open the 2025 queries."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from wdaoh.data import epa
from wdaoh.data.representations import build
from wdaoh.data.views import input_arm, observation_arm
from wdaoh.evaluation.metrics import errors, paired, summarize
from wdaoh.experiments.design import DESIGNS, REFERENCE, SELECTED, parameter_set
from wdaoh.models.training import Scale
from wdaoh.protocol import (
    GROUPS,
    KEYS,
    case_name,
    checked_sample,
    destination,
    digest,
    identity,
    settings,
    write_json,
)


@dataclass(frozen=True)
class Task:
    """A frozen scientific setting and its reader-owned input, model and prediction locations."""

    work: Path
    provider: str
    collection: str
    pollutant: str
    horizon: int
    seed: int
    model: str
    experiment: str
    weights: Path | None

    candidate: str | None = None

    @property
    def case(self):
        return case_name(
            self.provider, self.collection, self.pollutant, self.horizon, self.seed
        )

    @property
    def inputs(self):
        return self.work / "inputs" / self.case

    @property
    def output(self):
        return self.work / "forecasts" / self.case / self.model / self.experiment

    def sample(self, arm, group):
        directory = self.inputs / input_arm(arm)
        query = checked_sample(directory, group, self.horizon)
        return observation_arm(query, arm, self.pollutant)

    def config(self, arm):
        if self.candidate:
            return settings("model_parameters")[self.model][self.candidate]
        return parameter_set(
            self.model,
            self.pollutant,
            self.horizon,
            self.seed,
            arm,
            self.experiment,
            self.collection,
        )

    def arms(self):
        if self.experiment not in DESIGNS:
            raise ValueError("Unknown experiment")
        provider, models, arms = DESIGNS[self.experiment]
        if self.provider != provider or self.model not in models:
            raise ValueError(
                "This model and provider are outside the selected protocol"
            )
        if self.experiment == "graph" and (
            self.pollutant not in ("CO", "O3", "PM25") or self.horizon not in (6, 24)
        ):
            raise ValueError("Graph study uses CO/O3/PM25 at 6 or 24 hours")
        return arms


def fit_model(task, arm, train, directory):
    """Dispatch to the registered fitting procedure for this model and information arm."""
    if task.model == "CatBoost":
        from wdaoh.models import boosting

        raw = train["X"].ndim == 3
        return boosting.fit(
            train,
            task.config(arm),
            task.seed,
            raw,
            2 if raw or task.provider == "EEA" else 1,
            directory,
            1e-8 if task.provider == "EEA" else 1e-6,
        )
    if task.model in ("TabICL", "TabPFN"):
        from wdaoh.models import pretrained

        return pretrained.fit(task, arm, train, directory)
    if task.model == "LocalSAGE-TCN":
        from wdaoh.models import local_graph

        return local_graph.fit(task, arm, train, directory)
    from wdaoh.models import sequence

    return sequence.fit(task, arm, train, directory)


def predictor(task, arm, directory, stats):
    """Restore a predictor from its fitted state or from its verified external checkpoint."""
    if task.model == "CatBoost":
        from wdaoh.models import boosting

        return boosting.load(directory, stats)
    if task.model in ("TabICL", "TabPFN"):
        from wdaoh.models import pretrained

        return pretrained.load(task, arm, directory, stats)
    if task.model == "LocalSAGE-TCN":
        from wdaoh.models import local_graph

        return local_graph.load(task, arm, directory, stats)
    from wdaoh.models import sequence

    return sequence.load(task, arm, directory, stats)


def fit_and_validate(task, arm):
    """Score six 2024 populations. This stage has no 2025 sample loader calls."""
    base = task.inputs / input_arm(arm)
    manifest = json.loads((base / "samples.json").read_text())
    if (manifest["case"], manifest["arm"]) != (task.case, input_arm(arm)):
        raise ValueError(
            "Training sample identity differs from the requested case or view"
        )
    bindings = {
        k: v
        for k, v in manifest["files"].items()
        if k == "train" or k.startswith("validation_")
    }
    recipe = dict(
        case=task.case,
        model=task.model,
        arm=arm,
        parameters=task.config(arm),
        inputs=bindings,
    )
    folder = task.work / "fits" / task.case / task.model / identity(recipe)
    with destination(folder) as build:
        if build:
            train = task.sample(arm, "train")
            details, stats = fit_model(task, arm, train, build)
            np.savez(build / "scale.npz", **vars(stats))
            predict = predictor(task, arm, build, stats)
            metrics = {}
            for group in GROUPS:
                row = task.sample(arm, "validation_" + group)
                pred = predict(row) if len(row["y"]) else np.empty(0)
                metrics[group] = errors(row["y"], pred, stats.y_std)
                np.savez_compressed(
                    build / ("validation_" + group + ".npz"),
                    prediction=pred,
                    **{k: row[k] for k in (*KEYS, "y", "history_observed_count")},
                )
            artifacts = {p.name: digest(p) for p in build.iterdir() if p.is_file()}
            write_json(
                build / "validation.json",
                dict(
                    recipe=recipe,
                    details=details,
                    groups=metrics,
                    LMH=summarize(metrics),
                    artifacts=artifacts,
                ),
            )
    validated = json.loads((folder / "validation.json").read_text())
    if validated["recipe"] != recipe or any(
        digest(folder / name) != h for name, h in validated["artifacts"].items()
    ):
        raise ValueError("Changed validation fit or predictions")
    return folder, validated


def freeze(task, reselect=False):
    """Select only after all arms of this setting finish validation. Exact ties retain registered order."""
    rows = {}
    screened = {}
    for arm in task.arms():
        if reselect and task.model in settings("shortlists"):
            trials = {
                tag: fit_and_validate(replace(task, candidate=tag), arm)
                for tag in settings("shortlists")[task.model]
            }
            winner = min(trials, key=lambda tag: trials[tag][1]["LMH"]["MAE"])
            if winner != task.config(arm)["id"]:
                raise ValueError(
                    "Validation shortlist winner differs from the registered per-arm decision"
                )
            screened[arm] = {tag: r[1]["LMH"]["MAE"] for tag, r in trials.items()}
            rows[arm] = trials[winner]
        else:
            rows[arm] = fit_and_validate(task, arm)
    ref = next(iter(rows.values()))[0]
    for path, _ in list(rows.values())[1:]:
        for group in GROUPS:
            with (
                np.load(ref / ("validation_" + group + ".npz")) as a,
                np.load(path / ("validation_" + group + ".npz")) as b,
            ):
                paired(a, b)
    scores = {arm: r["LMH"]["MAE"] for arm, (_, r) in rows.items()}
    natural = {arm: r["groups"]["all_strata"]["MAE"] for arm, (_, r) in rows.items()}
    choices = {
        "sparse_equal3": min(scores, key=scores.get),
        "all_strata": min(natural, key=natural.get),
    }
    if task.experiment == "native-views":
        choices = {"sparse_equal3": choices["sparse_equal3"]}
    arms = (
        list(dict.fromkeys(choices.values()))
        if task.experiment in SELECTED
        else list(rows)
    )
    decision = dict(
        case=task.case,
        model=task.model,
        experiment=task.experiment,
        year=2024,
        objective="arithmetic mean of low, medium and high raw MAEs",
        scores=scores,
        natural_scores=natural,
        choices=choices if task.experiment in SELECTED else {},
        test_arms=arms,
        fits={a: str(p.relative_to(task.work)) for a, (p, _) in rows.items()},
        validation_sha256={
            a: digest(p / "validation.json") for a, (p, _) in rows.items()
        },
    )
    target = task.work / "decisions" / task.case / task.model / task.experiment
    with destination(target) as build:
        if build:
            write_json(build / "freeze.json", decision)
    if json.loads((target / "freeze.json").read_text()) != decision:
        raise ValueError("Conflicting validation decision")
    if screened:
        write_json(target / "shortlist.json", screened)
    return target / "freeze.json"


def forecast(task, decision_path):
    """Load frozen fits and predict held-out rows. Persist physical station identities and evaluation denominators."""
    decision = json.loads(decision_path.read_text())
    if (
        decision["case"],
        decision["model"],
        decision["experiment"],
        decision["year"],
    ) != (task.case, task.model, task.experiment, 2024):
        raise ValueError("Mismatched validation freeze")
    with destination(task.output) as build:
        if build:
            train = checked_sample(task.inputs / REFERENCE, "train", task.horizon)
            reference_std = max(float(train["y"].std()), 1e-6)
            arms = {}
            for arm in decision["test_arms"]:
                fit = task.work / decision["fits"][arm]
                if (
                    digest(fit / "validation.json")
                    != decision["validation_sha256"][arm]
                ):
                    raise ValueError("Changed validation decision input")
                fit_record = json.loads((fit / "validation.json").read_text())
                if any(
                    digest(fit / k) != v for k, v in fit_record["artifacts"].items()
                ):
                    raise ValueError(
                        "Changed fitted model, scaler or validation output"
                    )
                with np.load(fit / "scale.npz") as saved:
                    stats = Scale(
                        saved["mean"],
                        saved["deviation"],
                        float(saved["y_mean"]),
                        float(saved["y_std"]),
                        bool(saved["raw"]),
                    )
                predict = predictor(task, arm, fit, stats)
                manifest = json.loads(
                    (task.inputs / input_arm(arm) / "samples.json").read_text()
                )
                if (manifest["case"], manifest["arm"]) != (task.case, input_arm(arm)):
                    raise ValueError(
                        "Test sample identity differs from the frozen case or view"
                    )
                folder = build / arm
                folder.mkdir()
                files = {}
                for group in GROUPS:
                    row = task.sample(arm, "test_" + group)
                    pred = predict(row) if len(row["y"]) else np.empty(0)
                    if np.shape(pred) != row["y"].shape or not np.isfinite(pred).all():
                        raise ValueError("Invalid forecast")
                    path = folder / (group + ".npz")
                    np.savez_compressed(
                        path,
                        prediction=pred,
                        site=np.asarray(manifest["station_ids"])[row["site_index"]],
                        **{k: row[k] for k in (*KEYS, "y", "history_observed_count")},
                    )
                    files[group] = digest(path)
                arms[arm] = files
            units = epa.UNITS[task.pollutant] if task.provider == "EPA" else "µg/m³"
            write_json(
                build / "forecast.json",
                dict(
                    case=task.case,
                    model=task.model,
                    experiment=task.experiment,
                    arms=arms,
                    freeze_sha256=digest(decision_path),
                    reference_std=reference_std,
                    units=units,
                ),
            )
    if json.loads((task.output / "forecast.json").read_text())[
        "freeze_sha256"
    ] != digest(decision_path):
        raise ValueError("Previous forecasts reference a different freeze")
    print(f"2025 forecasts: {task.output}", flush=True)
    return task.output


def run(
    work, provider, collection, pollutant, horizon, seed, model, experiment, weights
):
    """Fit, freeze and evaluate eligible held-out queries for a single independently runnable case."""
    if experiment in (
        "nine-view",
        "native-views",
        "compact",
        "long-history",
        "raw-selection",
        "joint",
    ):
        raise ValueError(
            "Use study --execute for the complete model-wide validation freeze"
        )
    task = Task(
        work, provider, collection, pollutant, horizon, seed, model, experiment, weights
    )
    build(task)
    return forecast(task, freeze(task))
