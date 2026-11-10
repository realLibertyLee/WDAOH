"""Reduce hash-pinned EPA AirData files to fixed-monitor hourly matrices and training-only eligibility."""

import csv
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from wdaoh.data.providers import sources
from wdaoh.data.windows import HOUR, LENGTH, START
from wdaoh.protocol import CONFIG

CODES = {
    "CO": "42101",
    "SO2": "42401",
    "NO2": "42602",
    "O3": "44201",
    "PM25": "88101",
    "PM10": "81102",
}
UNITS = {
    "CO": "Parts per million",
    "SO2": "Parts per billion",
    "NO2": "Parts per billion",
    "O3": "Parts per million",
    "PM25": "Micrograms/cubic meter (LC)",
    "PM10": "Micrograms/cubic meter (25 C)",
}
STATES = {
    1,
    2,
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    53,
    54,
    55,
    56,
}
COLUMNS = [
    "State Code",
    "County Code",
    "Site Num",
    "POC",
    "Parameter Code",
    "Units of Measure",
    "Qualifier",
    "Method Type",
    "Method Code",
    "Time GMT",
    "Date GMT",
    "Sample Measurement",
    "Datum",
    "Latitude",
    "Longitude",
]


def verified_file(raw: Path, item: dict) -> Path:
    """Require a regular native file with the catalogued byte count and SHA before any observations enter a task."""
    path = raw / item["file"]
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != int(item["bytes"])
    ):
        raise ValueError("Missing or changed official vintage: " + item["id"])
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
            raise ValueError("Source SHA-256 differs: " + item["id"])
    return path


def partition(path, item, pollutant):
    """Return accepted monitor-hour rows; contradicting preferred method records remove that hour."""
    with (CONFIG / "epa_qualifiers.csv").open() as stream:
        allow = {"", "1V", "V", "MD", "7", "VB", "XS"} | {
            r[0] for r in csv.reader(stream) if {"INFORM", "REQEXC"}.intersection(r)
        }
    parts = []
    with zipfile.ZipFile(path) as archive, archive.open(item["member"]) as stream:
        for data in pd.read_csv(
            stream, usecols=COLUMNS, dtype=str, keep_default_na=False, chunksize=160000
        ):
            end = pd.to_datetime(
                data["Date GMT"] + " " + data["Time GMT"],
                format="%Y-%m-%d %H:%M",
                errors="coerce",
                utc=True,
            ) + pd.Timedelta(hours=1)
            value = pd.to_numeric(data["Sample Measurement"], errors="coerce")
            lat = pd.to_numeric(data.Latitude, errors="coerce")
            lon = pd.to_numeric(data.Longitude, errors="coerce")
            state = pd.to_numeric(data["State Code"], errors="coerce")
            ok = (
                (data["Parameter Code"] == CODES[pollutant])
                & (data["Units of Measure"] == UNITS[pollutant])
                & np.isfinite(value)
                & data.Qualifier.isin(allow)
                & data["Method Type"].isin(["FRM", "FEM"])
                & end.notna()
                & data["Time GMT"].str.fullmatch(r"\d\d:00")
                & state.isin(STATES)
                & (data.Datum == "WGS84")
                & lat.between(-90, 90)
                & lon.between(-180, 180)
            )
            if pollutant == "PM25":
                ok &= ~data["Method Code"].isin(["236", "238"])
            data = data.loc[ok]
            parts.append(
                pd.DataFrame(
                    dict(
                        site=(
                            state[ok].astype("int32") * 10000000
                            + data["County Code"].astype("int32") * 10000
                            + data["Site Num"].astype("int32")
                        ),
                        poc=data.POC.astype("int16"),
                        hour=(
                            (end[ok].dt.as_unit("ns").astype("int64") - START) // HOUR
                        ).astype("int32"),
                        value=value[ok].astype("float32"),
                        lat=lat[ok],
                        lon=lon[ok],
                        rank=data["Method Type"]
                        .map({"FRM": 0, "FEM": 1})
                        .astype("int8"),
                        method=pd.to_numeric(data["Method Code"], errors="coerce")
                        .fillna(-1)
                        .astype("int16"),
                    )
                )
            )
    keys = ["site", "poc", "hour"]
    rows = pd.concat(parts, ignore_index=True).sort_values(
        keys + ["rank", "method"], kind="stable"
    )
    dup = rows[rows.duplicated(keys, keep=False)]
    if len(dup):
        preferred = dup.drop_duplicates(keys)[keys + ["rank", "method"]]
        conflict = dup.groupby(keys + ["rank", "method"]).value.transform("nunique") > 1
        bad = dup[conflict].merge(preferred, on=keys + ["rank", "method"])
        rows = rows[
            ~pd.MultiIndex.from_frame(rows[keys]).isin(
                pd.MultiIndex.from_frame(bad[keys].drop_duplicates())
            )
        ]
    return rows.drop_duplicates(keys).drop(columns=["rank", "method"])


def network(raw: Path, pollutant: str):
    """Select each site's POC from 2021–23 coverage and stable coordinates; fill 2021–25 UTC completion hours."""
    frames = []
    for item in (
        r for r in sources() if r["domain"] == "EPA" and r["pollutant"] == pollutant
    ):
        path = verified_file(raw, item)
        frame = partition(path, item, pollutant)
        year = int(item["period"])
        use = (frame.hour.between(0, 35063) & (year <= 2024)) | (
            frame.hour.between(35064, LENGTH - 1) & (year >= 2024)
        )
        frames.append(frame.loc[use])
        print("EPA partition", pollutant, year, len(frame), flush=True)
    if len(frames) != 5:
        raise ValueError("Five annual source partitions are required")
    frame = pd.concat(frames, ignore_index=True)
    train = frame[frame.hour < 26280].copy()
    train["year"] = np.searchsorted([8760, 17520], train.hour, side="right") + 2021
    n = (
        train.groupby(["site", "poc", "year"])
        .hour.nunique()
        .unstack(fill_value=0)
        .reindex(columns=[2021, 2022, 2023], fill_value=0)
    )
    geo = train.groupby(["site", "poc"]).agg(
        lat=("lat", "median"),
        lon=("lon", "median"),
        a=("lat", "min"),
        b=("lat", "max"),
        c=("lon", "min"),
        d=("lon", "max"),
    )
    geo["min_annual"] = n.min(axis=1)
    geo["total"] = n.sum(axis=1)
    roster = (
        geo[(geo.b - geo.a <= 0.001) & (geo.d - geo.c <= 0.001) & (geo.min_annual > 0)]
        .reset_index()
        .sort_values(
            ["site", "min_annual", "total", "poc"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .drop_duplicates("site")
        .sort_values("site")
        .reset_index(drop=True)
    )
    chosen = frame.merge(
        roster[["site", "poc", "lat", "lon"]], on=["site", "poc"], suffixes=("", "_ref")
    )
    chosen = chosen[
        (abs(chosen.lat - chosen.lat_ref) <= 0.001)
        & (abs(chosen.lon - chosen.lon_ref) <= 0.001)
    ]
    keys = ["site", "hour"]
    if (
        chosen[chosen.duplicated(keys, keep=False)]
        .groupby(keys)
        .value.nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("Contradictory cross-partition measurement")
    chosen = chosen.drop_duplicates(keys)
    values = np.full((len(roster), LENGTH), np.nan, np.float32)
    values[
        chosen.site.map(dict(zip(roster.site, roster.index))).to_numpy(),
        chosen.hour.to_numpy(),
    ] = chosen.value
    roster["site"] = [
        f"{s // 10000000:02d}-{s // 10000 % 1000:03d}-{s % 10000:04d}"
        for s in roster.site
    ]
    return values, roster


def sample_sites(roster: pd.DataFrame, seed: int) -> np.ndarray:
    """Draw 100 (or 50) distinct sites from monitors with ≥876 eligible hours in each training year."""
    pool = np.flatnonzero(roster.min_annual.to_numpy() >= 876)
    if len(pool) < 50:
        raise ValueError("Fewer than 50 eligible training stations")
    return np.sort(
        np.random.default_rng(seed).choice(
            pool, 100 if len(pool) >= 100 else 50, replace=False
        )
    )
