"""Verified E1a concentrations, activity intervals and training-eligible collection membership."""

import decimal
import hashlib
import math

import numpy as np
import pandas as pd

from wdaoh.data import windows
from wdaoh.data.windows import GRID, HOUR, LENGTH, START
from wdaoh.protocol import identity, settings

CODES = {"CO": 10, "NO2": 8, "SO2": 1, "PM25": 6001, "PM10": 5, "O3": 7}


def eea_point(raw, item, spans_by_year):
    """Apply E1a quality and full-hour provenance to native measurements; return qualified µg/m³ and natural validity."""
    point = item["id"]
    frame = pd.read_parquet(raw / item["file"])
    if (
        set(frame.Samplingpoint) != {point}
        or set(frame.Pollutant) != {CODES[item["pollutant"]]}
        or frame.End.dt.tz is not None
    ):
        raise ValueError("Unexpected E1a identity or hourly clock: " + point)
    start = (frame.Start - pd.Timedelta(hours=1)).dt.tz_localize("UTC")
    end = (frame.End - pd.Timedelta(hours=1)).dt.tz_localize("UTC")
    unit = frame.Unit.map(
        {"ug.m-3": 1.0, "µg.m-3": 1.0, "μg.m-3": 1.0, "mg.m-3": 1000.0}
    )
    concentration = pd.to_numeric(frame.Value, errors="coerce").astype(float) * unit
    ok = (
        (frame.Verification == 1)
        & ((frame.Validity == 1) | ((frame.Pollutant == 7) & (frame.Validity == 4)))
        & (end - start == pd.Timedelta(hours=1))
        & (frame.AggType == "hour")
        & (end.dt.minute == 0)
        & (end.dt.second == 0)
        & np.isfinite(concentration)
        & (end >= GRID[0])
        & (end <= GRID[-1])
    )
    rows = pd.DataFrame(dict(end=end[ok], y=concentration[ok]))
    conflicts = rows.groupby("end").y.nunique()
    rows = (
        rows[~rows.end.isin(conflicts[conflicts > 1].index)]
        .drop_duplicates("end")
        .sort_values("end")
    )
    ticks = rows.end.dt.as_unit("ns").astype("int64").to_numpy()
    eligible = np.zeros(len(rows), dtype=bool)
    for year, spans in spans_by_year.items():
        in_year = rows.end.dt.year.to_numpy() == int(year)
        for first, last in spans:
            eligible |= in_year & (ticks - HOUR >= first) & (ticks <= last)
    index = (ticks - START) // HOUR
    natural = np.zeros(LENGTH, bool)
    natural[index] = True
    series = np.full(LENGTH, np.nan, np.float32)
    series[index[eligible]] = rows.y.to_numpy()[eligible]
    return series, natural


def quantum(geo, axis):
    """Return the original coordinate's precision in degrees, floored at 0.0001°."""
    raw = geo[2].split() if isinstance(geo[2], str) else geo[2]
    if geo[3].startswith("longitude"):
        raw = raw[::-1]
    try:
        return max(
            1e-4,
            float(
                decimal.Decimal(10)
                ** decimal.Decimal(str(raw[axis])).as_tuple().exponent
            ),
        )
    except (decimal.InvalidOperation, TypeError, IndexError):
        return 1e-4


def compatible(first, second):
    """Check coordinate agreement within the combined rounding precision of the two records."""
    return all(
        abs(first[k] - second[k])
        <= (quantum(first, k) + quantum(second, k)) / 2 + 1e-11
        for k in (0, 1)
    )


def distance(a, b):
    """Calculate great-circle distances in kilometres between station coordinates."""
    lat, lon, other_lat, other_lon = map(math.radians, [*a, *b])
    z = (
        math.sin((lat - other_lat) / 2) ** 2
        + math.cos(lat) * math.cos(other_lat) * math.sin((lon - other_lon) / 2) ** 2
    )
    return 12742000 * math.asin(math.sqrt(min(1, max(0, z))))


def eea_network(points, bank, variant, pollutant):
    """Freeze the station cap, same-station compatible suppliers and five neighbors from training eligibility."""
    ranked = []
    for key, (values, _) in bank.items():
        p = points[key]
        if variant == "MULTIYEAR_STABLE" and not p["stable"]:
            continue
        annual = [
            np.isfinite(values[windows.YEARS == year]).sum()
            for year in (2021, 2022, 2023)
        ]
        if min(annual) >= 876:
            ranked.append(
                (p["station"], p["pollutant"], -min(annual), -sum(annual), key)
            )
    suppliers = {}
    for station, pol, _, __, key in sorted(ranked):
        suppliers.setdefault((station, pol), key)
    eligible = [k for (_, p), k in suppliers.items() if p == pollutant]
    cap = sorted(
        eligible,
        key=lambda k: hashlib.sha256(points[k]["station_id"].encode()).digest(),
    )[:100]
    roster = sorted(cap, key=lambda k: points[k]["station_id"])
    geo = np.array([points[k]["geo"][:2] for k in roster])
    ds = np.full((len(geo), len(geo)), np.inf)
    for i in range(len(geo)):
        for j in range(i):
            ds[i, j] = ds[j, i] = distance(geo[i], geo[j])
    neighbors = np.argsort(ds, axis=1, kind="stable")[:, : min(5, len(geo) - 1)]
    names = []
    aux = np.full((len(roster), 5, windows.LENGTH), np.nan, np.float32)
    for i, k in enumerate(roster):
        p = points[k]
        slots = []
        for j, pol in enumerate(a for a in CODES if a != pollutant):
            key = suppliers.get((p["station"], pol))
            valid = bool(key and compatible(p["geo"], points[key]["geo"]))
            slots.append(key if valid else None)
            if valid:
                aux[i, j] = bank[key][0]
        names.append(slots)
    proof = dict(
        sites=len(roster),
        stations_sha=identity(roster),
        neighbors_sha=identity(neighbors.tolist()),
        suppliers_sha=identity(names),
    )
    if proof != settings("eea_topology")[variant + "/" + pollutant]:
        raise ValueError(
            "Frozen station/supplier/neighbor identities differ: "
            + variant
            + "/"
            + pollutant
        )
    proof["physical_sites"] = [points[k]["station_id"] for k in roster]
    values = np.stack([bank[k][0] for k in roster])
    natural = np.stack([bank[k][1] for k in roster])
    return (
        values,
        geo,
        aux,
        neighbors,
        windows.history_count(values),
        windows.history_count(np.where(natural, 0.0, np.nan)),
        proof,
    )
