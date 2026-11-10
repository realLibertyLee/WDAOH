"""Shared study identifiers and integrity checks for reader-generated artifacts."""

import contextlib
import fcntl
import gzip
import hashlib
import json
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parent / "config"
GROUPS = ("all_strata", "zero", "near_no_history", "low", "medium", "high")
LMH = ("low", "medium", "high")
KEYS = ("site_index", "origin_index", "target_index")
POLLUTANTS = ("CO", "NO2", "SO2", "PM25", "PM10", "O3")
SEEDS = (10, 20, 30, 40, 50)
HORIZONS = (6, 12, 24)


@lru_cache
def settings(name: str):
    """Read immutable reader configs without consulting an external data tree."""
    path = CONFIG / (name + ".json")
    if path.is_file():
        return json.loads(path.read_text())
    with gzip.open(path.with_suffix(".json.gz"), "rt") as stream:
        return json.load(stream)


def identity(value) -> str:
    """Hash a JSON-compatible configuration with a deterministic key order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def digest(path: Path) -> str:
    """Compute a streaming SHA-256 of a local artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    """Replace one result file atomically and reject nonfinite JSON values."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write("\n")
    Path(stream.name).replace(path)


@contextlib.contextmanager
def destination(path: Path):
    """Publish a complete stage atomically. A second invocation sees the completed directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / ("." + path.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            yield None
            return
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            build = Path(tmp) / "build"
            build.mkdir()
            yield build
            build.rename(path)


def case_name(provider, collection, pollutant, horizon, seed):
    """Identify a single domain, collection, pollutant, horizon and station seed."""
    return f"{provider}_{'PRIMARY' if provider == 'EPA' else collection}_{pollutant}_h{horizon}_seed{seed}"


def checked_sample(directory: Path, group: str, horizon: int) -> dict:
    """Load one matched sample with its filed checksum, year bounds and unique station-target keys."""
    manifest = json.loads((directory / "samples.json").read_text())
    return read_sample(directory, manifest, group, horizon)


def read_sample(directory, manifest, group, horizon):
    """Check native dimensions, population membership, UTC-year bounds and the recorded file digest."""
    path = directory / (group + ".npz")
    if digest(path) != manifest["files"][group]:
        raise ValueError(f"Changed sample file {path}")
    with np.load(path, allow_pickle=False) as data:
        fields = dict(data)
    if group == "normalizer":
        return fields
    stage = group.split("_", 1)[0]
    start, end = {
        "train": (0, 26280),
        "validation": (26280, 35064),
        "test": (35064, 43824),
    }[stage]
    site, origin, target = (fields[k] for k in KEYS)
    n = len(fields["y"])
    if (
        any(
            np.asarray(key).shape != (n,) or np.asarray(key).dtype.kind not in "iu"
            for key in (site, origin, target)
        )
        or np.any(site < 0)
        or np.any(site >= len(manifest["station_ids"]))
    ):
        raise ValueError(f"Invalid station or time index in {path}")
    if stage != "train":
        bounds = {
            "all_strata": (0, 24),
            "zero": (0, 0),
            "near_no_history": (1, 4),
            "low": (5, 11),
            "medium": (12, 17),
            "high": (18, 24),
        }
        lo, hi = bounds[group.split("_", 1)[1]]
        history = fields["history_observed_count"]
        if history.shape != (n,) or np.any((history < lo) | (history > hi)):
            raise ValueError(f"Support group membership changed in {path}")
    if (
        not np.isfinite(fields["y"]).all()
        or not np.all(target - origin == horizon)
        or len(set(zip(site, target))) != n
        or np.any(origin < start)
        or np.any(target >= end)
        or len(fields["X"]) != n
    ):
        raise ValueError(f"Invalid targets, time bounds or sample alignment in {path}")
    return fields
