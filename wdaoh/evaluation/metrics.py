"""Measure each real evaluation population and the equal-weight low/medium/high composite."""

import json
from pathlib import Path

import numpy as np

from wdaoh.protocol import GROUPS, KEYS, LMH, digest, write_json


def errors(y, prediction, target_std):
    """Compute physical-unit MAE/RMSE and training-SD-normalized errors in float64."""
    target = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if (
        target.ndim != 1
        or prediction.shape != target.shape
        or not np.isfinite(target).all()
        or not np.isfinite(prediction).all()
        or not np.isfinite(target_std)
        or target_std <= 0
    ):
        raise ValueError("Invalid prediction, target or training SD")
    n = len(target)
    if not n:
        return dict(n=0, MAE=None, RMSE=None, nMAE=None, nRMSE=None)
    residual = prediction - target
    mae = float(np.abs(residual).mean())
    rmse = float(np.sqrt(np.square(residual).mean()))
    return dict(n=n, MAE=mae, RMSE=rmse, nMAE=mae / target_std, nRMSE=rmse / target_std)


def summarize(groups):
    """Each LMH group contributes one third, including when actual row counts differ."""
    if any(groups[g]["n"] == 0 for g in LMH):
        raise ValueError("Incomplete LMH population")
    return {
        k: float(np.mean([groups[g][k] for g in LMH]))
        for k in ("MAE", "RMSE", "nMAE", "nRMSE")
    }


def paired(first, second):
    """Require identical ordered station, origin, target and measured label before any contrast."""
    fields = (
        (*KEYS, "y", "site") if "site" in first or "site" in second else (*KEYS, "y")
    )
    for key in fields:
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f"Unpaired forecasts: {key}")


def evaluate(directory: Path):
    """Read only this forecast run, verify all population artifacts and write physical-unit and normalized errors."""
    run = json.loads((directory / "forecast.json").read_text())
    result = {}
    for arm, files in run["arms"].items():
        groups = {}
        for group in GROUPS:
            path = directory / arm / (group + ".npz")
            if digest(path) != files[group]:
                raise ValueError("Changed prediction file: " + str(path))
            with np.load(path, allow_pickle=False) as row:
                groups[group] = errors(
                    row["y"], row["prediction"], run["reference_std"]
                )
        result[arm] = dict(groups=groups, LMH=summarize(groups))
        print(
            f"{run['case']} {arm} 2025 LMH MAE={result[arm]['LMH']['MAE']:.8g}, RMSE={result[arm]['LMH']['RMSE']:.8g}; Natural n={groups['all_strata']['n']}"
        )
    if len(result) > 1:
        ref = next(iter(result))
        differences = {}
        for arm in list(result)[1:]:
            for group in GROUPS:
                with (
                    np.load(directory / ref / (group + ".npz")) as a,
                    np.load(directory / arm / (group + ".npz")) as b,
                ):
                    paired(a, b)
            differences[arm + " minus " + ref] = {
                m: result[arm]["LMH"][m] - result[ref]["LMH"][m]
                for m in result[ref]["LMH"]
            }
    else:
        differences = {}
    metrics = dict(
        case=run["case"],
        model=run["model"],
        year=2025,
        arms=result,
        LMH_differences=differences,
        units=run["units"],
        training_target_std=run["reference_std"],
    )
    write_json(directory / "metrics.json", metrics)
    return metrics
