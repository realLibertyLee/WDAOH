"""Target-concentration tail diagnostics for the original matched R1 forecast cohorts."""

from collections import defaultdict

import numpy as np

from wdaoh.data.prepare import epa_network
from wdaoh.evaluation.comparisons import REPLICATES, interval
from wdaoh.evaluation.panels import decision, records, same_targets
from wdaoh.experiments.design import CONTEXT, REFERENCE
from wdaoh.protocol import LMH

# This ordering is part of the frozen per-pollutant bootstrap seed.
TAIL_ORDER = ("CO", "NO2", "O3", "PM10", "PM25", "SO2")
STRATA = ("all_strata", "low", "medium", "high", "zero")


def tail_contrasts(tasks_by_model):
    """Condition on training-network P95/P99. Compute pooled MAE or RMSE within strata and paired site intervals."""
    results = []
    for pol_index, pol in enumerate(TAIL_ORDER):
        for model, tasks in tasks_by_model.items():
            subset = [t for t in tasks if t.pollutant == pol]
            if len({(t.horizon, t.seed) for t in subset}) != 15 or len(subset) != 15:
                raise ValueError(
                    "Tail diagnostics require all three horizons and five seeds"
                )
            network, _ = epa_network(subset[0].work, pol)
            training = network[:, :26280]
            thresholds = np.quantile(
                training[np.isfinite(training)].astype(float),
                [0.95, 0.99],
                method="linear",
            )
            rows = defaultdict(list)
            for task in subset:
                freeze, run = decision(task)
                for g in STRATA:
                    a, b = [
                        records(task, freeze, run, arm, g, 2025)
                        for arm in (REFERENCE, CONTEXT)
                    ]
                    same_targets([a, b])
                    y = a["y"].astype(float)
                    rows[g].append(
                        (
                            a["site"],
                            a["origin_index"],
                            a["target_index"],
                            y,
                            abs(a["prediction"] - y),
                            abs(b["prediction"] - y),
                        )
                    )
            frame = np.unique(
                np.concatenate([v[0] for part in rows.values() for v in part])
            )
            draws = np.random.default_rng(
                np.random.SeedSequence([20261107, pol_index])
            ).multinomial(
                len(frame), np.full(len(frame), 1 / len(frame)), size=REPLICATES
            )
            weights = np.vstack((np.ones(len(frame)), draws))
            for level, threshold in zip((95, 99), thresholds):
                for population, strata in (
                    ("Natural", ("all_strata",)),
                    ("Zero", ("zero",)),
                    ("Balanced", LMH),
                ):
                    moments = []
                    eligible = True
                    memberships = 0
                    for g in strata:
                        site, origin, target, y, reference, context = (
                            np.concatenate(r) for r in zip(*rows[g])
                        )
                        mask = y >= threshold
                        idx = np.searchsorted(frame, site[mask])
                        memberships += int(mask.sum())
                        unique = len(set(zip(idx, origin[mask], target[mask])))
                        eligible &= unique >= 20 and len(np.unique(idx)) >= 2
                        denom = weights @ np.bincount(idx, minlength=len(frame))
                        errors = np.stack(
                            [
                                reference[mask],
                                context[mask],
                                reference[mask] ** 2,
                                context[mask] ** 2,
                            ],
                            axis=1,
                        )
                        sums = np.stack(
                            [
                                np.bincount(idx, weights=e, minlength=len(frame))
                                for e in errors.T
                            ],
                            axis=1,
                        )
                        with np.errstate(invalid="ignore", divide="ignore"):
                            moments.append((weights @ sums) / denom[:, None])
                    moment = np.mean(moments, axis=0)
                    for name, gain in (
                        ("MAE", moment[:, 0] - moment[:, 1]),
                        ("RMSE", np.sqrt(moment[:, 2]) - np.sqrt(moment[:, 3])),
                    ):
                        ci = interval(gain[1:])
                        valid = eligible and ci["valid"] == REPLICATES
                        results.append(
                            dict(
                                model=model,
                                pollutant=pol,
                                percentile=level,
                                training_threshold=float(threshold),
                                population=population,
                                metric=name,
                                memberships=memberships,
                                physical_sites_in_frame=len(frame),
                                positive_gain=float(gain[0])
                                if np.isfinite(gain[0])
                                else None,
                                interval=ci if valid else None,
                                valid_draws=ci["valid"],
                            )
                        )
    return results
