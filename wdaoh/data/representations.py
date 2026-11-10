"""Build matched engineered, nested-support and native-sequence alternatives from official EPA station hours."""

import numpy as np

from wdaoh.data import epa, features, history, windows
from wdaoh.data.context import context_mapper, nested_keys
from wdaoh.data.prepare import epa_network, save_samples
from wdaoh.experiments.design import CONTEXT, NINE, REFERENCE
from wdaoh.models.scaling import fit_stats, norm
from wdaoh.protocol import GROUPS, KEYS, checked_sample, settings


def raw_names(channels):
    """Name target-last flattened lag columns in chronological model-input order."""
    return [
        f"value_lag{23 - t}" if c == channels - 1 else f"raw_t{t}_c{c}"
        for t in range(24)
        for c in range(channels)
    ]


def store(task, arm, groups, sites, names=None):
    """Commit one matched view with its feature order and physical-axis identity."""
    save_samples(task.inputs / arm, groups, sites, arm, task.case, names)


def build(task):
    """Create only the requested study's alternatives. Fit all feature and target scales on 2021–23 support."""
    arms = task.arms()
    if not any(
        task.experiment == e
        for e in ("budget", "nine-view", "long-history", "native-views", "temporal")
    ):
        return
    if all((task.inputs / arm / "samples.json").is_file() for arm in arms):
        return
    bank = {
        p: epa_network(task.work, p) for p in ("CO", "NO2", "SO2", "PM25", "PM10", "O3")
    }
    values, stations = bank[task.pollutant]
    sites = stations.site.tolist()
    roster = epa.sample_sites(stations, task.seed)
    geo = stations[["lat", "lon"]].to_numpy()
    groups = (
        "train",
        *(f"{stage}_{g}" for stage in ("validation", "test") for g in GROUPS),
    )
    raw = {g: checked_sample(task.inputs / REFERENCE, g, task.horizon) for g in groups}
    ctx = context_mapper(bank, task.pollutant, roster)
    if task.experiment == "budget":
        build_budgets(task, raw, values, roster, geo, sites, ctx)
    elif task.experiment == "native-views":
        for arm in arms:
            cols = settings("native_channels")[arm]
            arrays = {}
            for g, q in raw.items():
                x = (
                    history.feature_long(
                        values, geo, q["site_index"], q["origin_index"]
                    )
                    if cols is None
                    else ctx(q)["X"][:, :, cols]
                )
                if cols is None and not np.array_equal(
                    x[:, -24:], q["X"], equal_nan=True
                ):
                    raise ValueError("Long-history last-24 identity")
                arrays[g] = dict(q, X=x)
            store(task, arm, arrays, sites)
    else:
        build_features(task, bank, roster, geo, sites, ctx, raw, arms)


def build_budgets(task, raw, values, roster, geo, sites, ctx):
    """Construct nested 1k, 5k and 10k training supports and transform the same evaluation keys."""
    train = raw["train"]
    small, (site, origin) = nested_keys(
        values, roster, train, task.pollutant, task.seed, task.horizon
    )
    target = origin + task.horizon
    extra = dict(
        X=windows.history_matrix(values, geo, site, origin),
        y=values[site, target],
        site_index=site,
        origin_index=origin,
        target_index=target,
    )
    extra["history_observed_count"] = np.isfinite(extra["X"][:, :, -1]).sum(1)
    both = {
        k: np.concatenate([train[k], extra[k]])
        for k in ("X", "y", *KEYS, "history_observed_count")
    }
    ordered = np.lexsort((both["site_index"], both["target_index"]))
    pools = {
        1000: {k: v[small] for k, v in train.items()},
        5000: train,
        10000: {k: v[ordered] for k, v in both.items()},
    }
    for n, support in pools.items():
        for arm in (REFERENCE, CONTEXT):
            group_inputs = {**raw, "train": support}
            if arm == CONTEXT:
                group_inputs = {g: ctx(q) for g, q in group_inputs.items()}
            tr = group_inputs["train"]
            st = fit_stats(tr["X"], tr["y"])
            arrays = {
                g: dict(
                    q, X=norm(q["X"], st).reshape(len(q["y"]), 24 * q["X"].shape[-1])
                )
                for g, q in group_inputs.items()
            }
            store(task, f"{arm}@{n}", arrays, sites, raw_names(tr["X"].shape[-1]))


def build_features(task, bank, roster, geo, sites, ctx, raw, arms):
    """Compute the registered tabular views from common support and query windows."""
    aux = [k.upper() for k in settings("feature_auxiliaries")[task.pollutant.lower()]]
    window = history.window_reader(bank, task.pollutant, roster, aux)
    train = raw["train"]
    support = window(train)
    scalers = {
        "target": features.train_scaler([support["target"], support["neighbors"]])
    }
    scalers.update({k: features.train_scaler([v]) for k, v in support["aux"].items()})
    missing = 1 - np.isfinite(support["target"][:, :24]).mean(axis=1)
    rates = {
        s: float(missing[train["site_index"] == s].mean())
        for s in np.unique(train["site_index"])
    }
    global_rate = float(missing.mean())
    scales = {
        False: fit_stats(train["X"], train["y"]),
        True: fit_stats(ctx(train)["X"], train["y"]),
    }
    outputs = {arm: {} for arm in arms}
    columns = {}
    for g, q in raw.items():
        site = q["site_index"]
        frame, blocks = features.features(
            support if g == "train" else window(q),
            q["origin_index"],
            task.horizon,
            geo[site],
            np.array([rates.get(s, global_rate) for s in site]),
            global_rate,
            scalers,
        )
        for arm in arms:
            if arm in NINE:
                names = [
                    c for block in settings("view_blocks")[arm] for c in blocks[block]
                ]
                x = frame[names].to_numpy()
            else:
                full = arm in ("C_S1", "D_S1_H")
                raw_x = ctx(q)["X"] if full else q["X"]
                names = raw_names(raw_x.shape[-1])
                x = norm(raw_x, scales[full]).reshape(len(site), len(names))
                if arm in ("B_S0_H", "D_S1_H"):
                    extra = blocks["T"] + blocks["MT"]
                    names = names + extra
                    x = np.concatenate([x, frame[extra].to_numpy()], axis=1)
            columns[arm] = names
            outputs[arm][g] = dict(q, X=x)
    for arm in arms:
        store(task, arm, outputs[arm], sites, columns[arm])
