"""Offline regression checks. The hourly cycle is a synthetic two-station fixture."""

import hashlib
import http.client
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from wdaoh.data.history import causal
from wdaoh.data.prepare import save_samples
from wdaoh.data.providers import finish_short_read
from wdaoh.data.views import observation_arm
from wdaoh.evaluation.comparisons import site_contrast, task_intervals
from wdaoh.evaluation.metrics import errors, evaluate, paired, summarize
from wdaoh.experiments.design import REFERENCE
from wdaoh.experiments.runner import Task, fit_and_validate, forecast, freeze
from wdaoh.experiments.study import settings_grid
from wdaoh.models.training import chronological_split
from wdaoh.protocol import GROUPS, checked_sample

CYCLE = np.genfromtxt(
    Path(__file__).parent / "fixtures/hourly_cycle.csv", delimiter=",", skip_header=1
)[:, 1].astype("float32")
COUNTS = dict(all_strata=8, zero=0, near_no_history=2, low=8, medium=14, high=22)


def samples(origins, count):
    """Create observed labels six hours ahead and mask older measurements before model input."""
    site = origins % 2
    times = origins[:, None] + np.arange(-23, 1)
    values = CYCLE[times % 24].copy() + site[:, None] * np.float32(0.25)
    observed = np.broadcast_to(np.arange(24) >= 24 - count, values.shape)
    x = np.stack((observed, np.where(observed, values, np.nan)), axis=2).astype(
        "float32"
    )
    target = origins + 6
    return dict(
        X=x,
        y=(CYCLE[target % 24] + site * 0.25).astype("float32"),
        site_index=site,
        origin_index=origins,
        target_index=target,
        history_observed_count=observed.sum(1),
    )


class PipelineTests(unittest.TestCase):
    def test_provider_confirmed_ranges(self):
        """Preserve positive short bodies and accept only the specified offsets and full vintage length."""
        source = bytes(range(256)) * 512
        stream = io.BytesIO(source[:10])
        stream.seek(0, 2)
        digest = hashlib.sha256(source[:10])
        seen = []

        def open_range(request, timeout):
            start, end = map(int, request.get_header("Range")[6:].split("-"))
            seen.append(start)
            response = io.BytesIO(source[start : end + 1])
            response.headers = {"Content-Range": f"bytes {start}-{end}/{len(source)}"}
            response.status = 206
            if len(seen) == 1:
                response.read = Mock(
                    side_effect=http.client.IncompleteRead(source[start : start + 7], 9)
                )
            return response

        host = Mock()
        host.open.side_effect = open_range
        url = "https://example.org/fixture"
        size, requests = finish_short_read(
            host, url, stream, 10, len(source), 1, digest
        )
        self.assertEqual((size, requests), (len(source), len(seen)))
        self.assertEqual(seen[:2], [10, 17])
        self.assertEqual(stream.getvalue(), source)
        self.assertEqual(digest.digest(), hashlib.sha256(source).digest())
        wrong_range = io.BytesIO(b"x")
        wrong_range.status = 206
        wrong_range.headers = {"Content-Range": "bytes 1-1/2"}
        host.open.side_effect = [wrong_range]
        with self.assertRaisesRegex(ValueError, "range and length"):
            finish_short_read(host, url, io.BytesIO(), 0, 2, 1, hashlib.sha256())

    def test_forecast_and_measure(self):
        """Exercise the real estimator, freeze, held-out labels, artifact integrity and cache reuse."""
        with tempfile.TemporaryDirectory() as tmp:
            task = Task(
                Path(tmp), "EPA", "PRIMARY", "CO", 6, 10, "CatBoost", "reference", None
            )
            arrays = {"train": samples(np.arange(60, 284), 20)}
            for stage, begin in (("validation", 26280), ("test", 35064)):
                for group in GROUPS:
                    arrays[f"{stage}_{group}"] = samples(
                        np.arange(begin + 24, begin + 30), COUNTS[group]
                    )
            save_samples(
                task.inputs / REFERENCE,
                arrays,
                ["fixture-west", "fixture-east"],
                REFERENCE,
                task.case,
            )
            fit, stop, _ = chronological_split(arrays["train"])
            self.assertGreater(
                arrays["train"]["origin_index"][stop].min()
                - arrays["train"]["target_index"][fit].max(),
                48,
            )
            # Validation must still finish when the test tensor files cannot be opened.
            held_out = list((task.inputs / REFERENCE).glob("test_*.npz"))
            for path in held_out:
                path.rename(path.with_suffix(".withheld"))
            decision = freeze(task)
            for path in held_out:
                path.with_suffix(".withheld").rename(path)
            frozen = json.loads(decision.read_text())
            self.assertEqual(frozen["test_arms"], [REFERENCE])
            folder = task.work / frozen["fits"][REFERENCE]
            with np.load(folder / "scale.npz") as scale:
                self.assertEqual(
                    float(scale["y_mean"]), float(arrays["train"]["y"].mean())
                )
                self.assertEqual(
                    float(scale["y_std"]), float(arrays["train"]["y"].std())
                )
            checked = fit_and_validate(task, REFERENCE)[0]
            self.assertEqual(checked, folder)
            result = evaluate(forecast(task, decision))
            with np.load(task.output / REFERENCE / "low.npz") as z:
                np.testing.assert_equal(z["y"], arrays["test_low"]["y"])
                np.testing.assert_equal(
                    z["site"],
                    np.array(["fixture-west", "fixture-east"])[
                        arrays["test_low"]["site_index"]
                    ],
                )
                mae = np.abs(
                    z["prediction"].astype(float) - z["y"].astype(float)
                ).mean()
            self.assertEqual(result["arms"][REFERENCE]["groups"]["low"]["MAE"], mae)
            self.assertTrue(np.isfinite(result["arms"][REFERENCE]["LMH"]["RMSE"]))
            self.assertAlmostEqual(
                result["arms"][REFERENCE]["LMH"]["MAE"], 0.21247526, places=7
            )
            (task.output / REFERENCE / "zero.npz").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "Changed prediction"):
                evaluate(task.output)
            (folder / "model.cbm").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "Changed validation fit"):
                fit_and_validate(task, REFERENCE)

    def test_time_and_physical_alignment(self):
        history = np.arange(10, dtype=float).reshape(1, 10)
        base = causal(history, np.array([0, -1]), np.array([2, 3]), 4)
        np.testing.assert_equal(base[0], [2.0, 1.0, 0.0, np.nan])
        self.assertTrue(np.isnan(base[1]).all())
        history[0, 3:] = 1000
        np.testing.assert_equal(
            causal(history, np.array([0, -1]), np.array([2, 3]), 4), base
        )
        group = samples(np.arange(35100, 35106), 14)
        reordered = {k: np.flip(v, axis=0) for k, v in group.items()}
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            paired(group, reordered)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "medium"
            save_samples(
                folder,
                {"test_medium": group, "test_zero": group},
                ["a", "b"],
                "toy",
                "toy",
            )
            checked_sample(folder, "test_medium", 6)
            with self.assertRaisesRegex(ValueError, "Support group"):
                checked_sample(folder, "test_zero", 6)
            with self.assertRaisesRegex(ValueError, "time bounds"):
                checked_sample(folder, "test_medium", 12)

    def test_stratum_weights_and_masked_sources(self):
        metrics = {
            g: errors(np.zeros(n), np.full(n, value), 2.0)
            for g, n, value in (("low", 1, 0), ("medium", 2, 3), ("high", 5, 9))
        }
        self.assertEqual(summarize(metrics)["MAE"], 4.0)
        self.assertEqual(summarize(metrics)["nMAE"], 2.0)
        self.assertIsNone(errors([], [], 1.0)["MAE"])
        with self.assertRaises(ValueError):
            errors([np.nan], [1.0], 1.0)
        x = np.ones((2, 24, 30), "float32")
        q = observation_arm({"X": x}, "P1_00", "CO")["X"]
        self.assertTrue(np.isnan(q[:, :, 9:29:2]).all())
        self.assertTrue((q[:, :, 10:29:2] == 0).all())
        np.testing.assert_equal(q[:, :, -1], x[:, :, -1])
        self.assertTrue((x == 1).all())

    def test_paired_cluster_estimators_and_scope(self):
        """Station reweighting and task resampling preserve a constant paired intervention effect."""
        x = task_intervals(np.ones((18, 5)))
        self.assertAlmostEqual(x["mean"], 1.0)
        self.assertAlmostEqual(x["task_ci"]["lower"], 1.0)
        sites = np.array(["a", "b"])
        count = np.array([1, 3])
        absolute = np.array([[4.0, 12.0], [2.0, 6.0]])
        squared = np.array([[16.0, 48.0], [4.0, 12.0]])
        result = site_contrast(
            [(sites, count, absolute, squared, 2.0)],
            [1, -1],
            np.random.default_rng(17),
            physical=True,
        )
        for metric in ("nMAE", "nRMSE"):
            self.assertAlmostEqual(result["metrics"][metric]["value"], 1.0)
            self.assertAlmostEqual(result["metrics"][metric]["ci"]["upper"], 1.0)
        self.assertEqual(result["metrics"]["MAE"]["value"], 2.0)
        with self.assertRaises(ValueError):
            task_intervals(np.ones((17, 5)))
        self.assertEqual(
            len(settings_grid(Path("."), "graph", "LocalSAGE-TCN", None)), 30
        )
        self.assertEqual(
            len(settings_grid(Path("."), "eea-selection", "CatBoost", None)), 180
        )
