"""Enumerate and run the study in dependency order, with complete model-specific validation before testing."""

import json
from itertools import product

from wdaoh.data.download import download
from wdaoh.data.prepare import prepare
from wdaoh.data.representations import build
from wdaoh.evaluation.comparisons import factorial_tables, r1_tables, selected_tables
from wdaoh.evaluation.metrics import evaluate
from wdaoh.evaluation.panels import (
    factorial_panel,
    fixed_validation_panel,
    graph_panels,
    selected_contrast,
    winner_frequencies,
)
from wdaoh.evaluation.tails import tail_contrasts
from wdaoh.experiments.design import DESIGNS, REFERENCE
from wdaoh.experiments.runner import Task, forecast, freeze
from wdaoh.models.upstream import acquire_sources
from wdaoh.protocol import HORIZONS, POLLUTANTS, SEEDS, digest, write_json


def settings_grid(work, experiment, model, weights):
    """Enumerate the registered collections, pollutants, horizons and five seeds for one model."""
    provider, models, _ = DESIGNS[experiment]
    if model not in models:
        raise ValueError("Model outside this experiment")
    collections = (
        ("STANDARD", "MULTIYEAR_STABLE") if provider == "EEA" else ("PRIMARY",)
    )
    pollutants, horizons = (
        (("PM25", "O3", "CO"), (6, 24))
        if experiment == "graph"
        else (POLLUTANTS, HORIZONS)
    )
    return [
        Task(work, provider, col, p, h, s, model, experiment, weights)
        for col, p, h, s in product(collections, pollutants, horizons, SEEDS)
    ]


def prepare_case(task):
    """Construct shared native inputs and the information representations needed by one setting."""
    context = task.provider == "EPA" and task.experiment != "reference"
    required = task.inputs / ("S1_CONTEXT" if context else REFERENCE) / "samples.json"
    if not required.is_file():
        prepare(
            task.work,
            task.provider,
            task.collection,
            task.pollutant,
            task.horizon,
            task.seed,
            context,
        )
    build(task)


def study(work, execute, experiment, model, weights, stage="all"):
    """Write the study schedule or execute its stages with a model-wide 2024 freeze before 2025."""
    work.mkdir(parents=True, exist_ok=True)
    experiments = (
        [experiment] if experiment else [k for k in DESIGNS if k != "reference"]
    )
    if any(k not in DESIGNS for k in experiments):
        raise ValueError("Unknown study experiment")
    schedule = {}
    for name in experiments:
        domain, models, arms = DESIGNS[name]
        chosen = (
            [model] if model and model in models else list(models) if not model else []
        )
        for m in chosen:
            grid = settings_grid(work, name, m, weights)
            schedule[name, m] = grid
    if not schedule:
        raise ValueError("No in-scope study settings")
    plan = []
    for (name, m), tasks in schedule.items():
        plan.append(
            dict(
                experiment=name,
                model=m,
                provider=tasks[0].provider,
                arms=DESIGNS[name][2],
                cases=[t.case for t in tasks],
                source="raw from the provider catalog",
                train="2021–23 frozen label budgets",
                validation="six population predictions and decisions/*/freeze.json",
                prediction="2025 per-arm NPZ files by support group in forecasts/",
                evaluation="metrics.json and cohort comparisons",
                stage_order=[
                    "acquire",
                    "prepare",
                    "validate",
                    "predict",
                    "evaluate",
                    "compare",
                ],
            )
        )
    for entry in plan:
        entry["compare_requires"] = (
            ["joint target-only forecasts for the same model and cases"]
            if entry["experiment"] == "raw-selection"
            else [
                "selected CatBoost forecasts on identical EEA collections for paired model effects"
            ]
            if entry["experiment"] == "eea-selection" and entry["model"] != "CatBoost"
            else []
        )
    write_json(work / "study.json", plan)
    print(
        f"{len(plan)} model/protocol schedules, {sum(len(ts) for ts in schedule.values())} settings in {work / 'study.json'}",
        flush=True,
    )
    if not execute:
        return
    if stage in ("all", "acquire"):
        for provider in sorted({ts[0].provider for ts in schedule.values()}):
            download(provider, "all", work / "raw")
        for m in sorted(
            {m for _, m in schedule} & {"DLinear+side", "ModernTCN", "iTransformer"}
        ):
            acquire_sources(work, m)
    if stage in ("all", "prepare"):
        for tasks in schedule.values():
            for task in tasks:
                prepare_case(task)
    for (name, m), tasks in schedule.items():
        release = work / "decisions" / "models" / name / m
        if stage in ("all", "validate"):
            decisions = {}
            for task in tasks:
                chosen = freeze(task, reselect=name == "joint")
                decisions[task.case] = digest(chosen)
            release.mkdir(exist_ok=True, parents=True)
            write_json(release / "model.json", decisions)
        if stage in ("all", "predict"):
            barrier = json.loads((release / "model.json").read_text())
            if set(barrier) != {t.case for t in tasks}:
                raise ValueError("Incomplete 2024 model-wide freeze")
            for task in tasks:
                chosen = work / "decisions" / task.case / m / name / "freeze.json"
                if digest(chosen) != barrier[task.case]:
                    raise ValueError("Changed frozen choice")
            for task in tasks:
                forecast(
                    task, work / "decisions" / task.case / m / name / "freeze.json"
                )
        if stage in ("all", "evaluate"):
            for task in tasks:
                evaluate(task.output)
        if stage in ("all", "compare"):
            output = work / "comparisons" / name / m
            output.mkdir(parents=True, exist_ok=True)
            for collection in sorted({t.collection for t in tasks}):
                current = [t for t in tasks if t.collection == collection]
                if name == "graph":
                    result = graph_panels(current)
                elif name in ("sources", "compact", "components"):
                    result = factorial_tables(current, name, m)
                else:
                    result = selected_tables(current)
                write_json(output / (collection + ".json"), result)
                if name in ("sources", "components", "compact"):
                    write_json(
                        output / (collection + "_global_site.json"),
                        factorial_panel(current, name),
                    )
                if name in (
                    "raw-selection",
                    "nine-view",
                    "long-history",
                    "native-views",
                    "eea-selection",
                ):
                    write_json(
                        output / (collection + "_winner_frequencies.json"),
                        winner_frequencies(current),
                    )
                if name in ("budget", "temporal"):
                    write_json(
                        output / (collection + "_validation.json"),
                        fixed_validation_panel(current),
                    )
                if name == "raw-selection":
                    write_json(
                        output / (collection + "_vs_target_history.json"),
                        selected_contrast(current),
                    )

    if stage in ("all", "compare") and "joint" in experiments:
        models = {m: tasks for (e, m), tasks in schedule.items() if e == "joint"}
        if len(models) == len(DESIGNS["joint"][1]):
            write_json(
                work / "comparisons/joint/target_concentration_tails.json",
                tail_contrasts(models),
            )
            write_json(
                work / "comparisons/joint/physical_site_intervals.json",
                r1_tables(models),
            )
    if stage in ("all", "compare"):
        for (name, m), tasks in schedule.items():
            if name == "eea-selection" and m != "CatBoost":
                for collection in ("STANDARD", "MULTIYEAR_STABLE"):
                    current = [t for t in tasks if t.collection == collection]
                    write_json(
                        work
                        / "comparisons"
                        / name
                        / m
                        / (collection + "_vs_CatBoost.json"),
                        selected_contrast(current, "CatBoost"),
                    )
