"""The chronological inner fit and refit on the complete, frozen label budget."""

from dataclasses import dataclass

import numpy as np

from wdaoh.models.scaling import fit_stats, norm


def chronological_split(train):
    """Split by target time, with a 48-hour purge before the first stopping origin."""
    order = np.argsort(train["target_index"], kind="stable")
    boundary = train["target_index"][order[int(0.8 * len(order))]]
    fit = order[train["target_index"][order] < boundary - 48]
    stop = order[train["origin_index"][order] >= boundary]
    if (
        not len(fit)
        or not len(stop)
        or train["target_index"][fit].max() >= train["origin_index"][stop].min() - 48
    ):
        raise ValueError("No purged train/stop split")
    return fit, stop, int(boundary)


@dataclass
class Scale:
    """Training-only input scale and physical-unit target centering for a single fit."""

    mean: np.ndarray
    deviation: np.ndarray
    y_mean: float
    y_std: float
    raw: bool

    def inputs(self, x):
        """Apply the fitted channel scale to raw sequences or retain precomputed training-scaled features."""
        if not self.raw:
            return x
        return norm(x, (self.mean, self.deviation))

    def labels(self, y):
        return (y - self.y_mean) / self.y_std

    def restore(self, prediction):
        return np.asarray(prediction).reshape(-1) * self.y_std + self.y_mean


def scale(x, y, raw=True, floor=1e-6):
    """Fit sequence channel statistics or retain a prepared matrix and fit only the target scale."""
    if raw:
        mean, deviation, center, spread = fit_stats(x, y)
        return Scale(mean, deviation, center, spread, True)
    return Scale(
        np.empty(0, np.float32),
        np.empty(0, np.float32),
        float(y.mean()),
        max(float(y.std()), floor),
        False,
    )
