"""Build reader-owned samples from verified monitor-hour observations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wdaoh.data import eea, epa, windows
from wdaoh.data.context import context_mapper, graph
from wdaoh.data.providers import sources
from wdaoh.protocol import case_name, destination, digest, settings, write_json


def epa_network(work: Path, pollutant: str):
    """Cache the train-selected POC roster and native hourly observations shared by all horizons and seeds."""
    path = work / "networks" / pollutant
    with destination(path) as build:
        if build:
            values, roster = epa.network(work / "raw", pollutant)
            physical = roster[["site", "poc", "lat", "lon"]].values.tolist()
            if physical != settings("epa_networks")[pollutant]["network"]:
                raise ValueError(
                    f"Monitor or coordinate identity changed for {pollutant}"
                )
            np.save(build / "hours.npy", values)
            roster.to_csv(build / "stations.csv", index=False)
            write_json(
                build / "sources.json",
                {
                    r["id"]: r["sha256"]
                    for r in sources()
                    if r["domain"] == "EPA" and r["pollutant"] == pollutant
                },
            )
            write_json(
                build / "network.json",
                {n: digest(build / n) for n in ("hours.npy", "stations.csv")},
            )
    expected = json.loads((path / "network.json").read_text())
    if any(digest(path / name) != value for name, value in expected.items()):
        raise ValueError(f"Changed hourly network for {pollutant}")
    return np.load(path / "hours.npy", mmap_mode="r"), pd.read_csv(
        path / "stations.csv", dtype={"site": str}, float_precision="round_trip"
    )


def eea_observations(work: Path):
    """Apply the frozen E1a activity, unit and clock rules to each eligible sampling point once."""
    path = work / "networks/EEA"
    points = settings("eea_points")
    with destination(path) as build:
        if build:
            files = {}
            for item in sources():
                if item["domain"] != "EEA" or item["id"] not in points:
                    continue
                epa.verified_file(work / "raw", item)
                series, natural = eea.eea_point(
                    work / "raw", item, points[item["id"]]["activity"]
                )
                filename = item["sha256"] + ".npz"
                np.savez_compressed(build / filename, values=series, natural=natural)
                files[item["id"]] = [filename, digest(build / filename)]
            if set(files) != set(points):
                raise ValueError("Incomplete EEA sampling-point catalog")
            write_json(build / "points.json", files)
    bank = {}
    for key, (filename, expected) in json.loads(
        (path / "points.json").read_text()
    ).items():
        if digest(path / filename) != expected:
            raise ValueError("Changed EEA point " + key)
        with np.load(path / filename, allow_pickle=False) as point:
            bank[key] = point["values"], point["natural"]
    return points, bank


def save_samples(path, arrays, sites, variant, case, columns=None):
    """Save arrays by stage and population with their physical station axis, column order and checksums."""
    with destination(path) as build:
        if build:
            files = {}
            if columns is not None:
                write_json(build / "columns.json", columns)
            for group, fields in arrays.items():
                target = build / (group + ".npz")
                np.savez_compressed(target, **fields)
                files[group] = digest(target)
            write_json(
                build / "samples.json",
                dict(case=case, arm=variant, station_ids=sites, files=files),
            )


def prepare(work, provider, collection, pollutant, horizon, seed, context=False):
    """Construct training and disjoint-year queries. S0 and S1 share ordered station-origin-target triples."""
    case = case_name(provider, collection, pollutant, horizon, seed)
    path = work / "inputs" / case
    if provider == "EPA":
        values, stations = epa_network(work, pollutant)
        roster = epa.sample_sites(stations, seed)
        sites = stations.site.tolist()
        if (
            stations.site.iloc[roster].tolist()
            != settings("epa_networks")[pollutant]["seed_sites"][str(seed)]
        ):
            raise ValueError("Frozen station draw mismatch")
        primary = windows.epa_payloads(
            values, stations[["lat", "lon"]].to_numpy(), roster, seed, horizon
        )
        save_samples(path / "S0_TARGET24", primary, sites, "S0_TARGET24", case)
        if context:
            bank = {
                k: (values, stations) if k == pollutant else epa_network(work, k)
                for k in eea.CODES
            }
            add = context_mapper(bank, pollutant, roster)
            save_samples(
                path / "S1_CONTEXT",
                {k: add(v) for k, v in primary.items()},
                sites,
                "S1_CONTEXT",
                case,
            )
            with destination(path / "graph") as build:
                if build:
                    np.savez_compressed(
                        build / "topology.npz", **graph(stations, roster)
                    )
    else:
        points, bank = eea_observations(work)
        values, geo, aux, neighbors, counts, natural, proof = eea.eea_network(
            points, bank, collection, pollutant
        )
        picked = windows.eea_groups(
            values, counts, natural, seed, eea.CODES[pollutant], horizon
        )
        for arm, arrays in windows.eea_payloads(
            values, geo, aux, neighbors, counts, natural, picked, horizon, seed == 10
        ):
            save_samples(path / arm, arrays, proof["physical_sites"], arm, case)
    print(f"Prepared {case} in {path}", flush=True)
    return path
