"""Frozen in-context regressors using separately obtained, checksum-bound provider weights."""

import os
from importlib.metadata import version

import numpy as np

from wdaoh.models.training import scale
from wdaoh.protocol import digest, settings


def weights(task):
    """Verify provider weight size, SHA-256 and the required package version before use."""
    spec = settings("pretrained")[task.model]
    if task.weights is None:
        raise ValueError(
            f"{task.model} needs {spec['checkpoint_filename']} from {spec['source_url']} under the provider terms"
        )
    path = task.weights / spec["checkpoint_filename"]
    if (
        not path.is_file()
        or path.stat().st_size != spec["checkpoint_bytes"]
        or digest(path) != spec["checkpoint_sha256"]
    ):
        raise ValueError("Missing or mismatched pretrained weights: " + str(path))
    if version(spec["package"]) != spec["version"]:
        raise ValueError(f"{spec['package']} {spec['version']} is required")
    return path


def fit(task, arm, train, directory):
    """Bind the support and provider checkpoint; the estimator consumes this exact context on each new process."""
    path = weights(task)
    st = scale(train["X"], train["y"], train["X"].ndim == 3)
    return {"external_checkpoint": digest(path), "support_rows": len(train["y"])}, st


def load(task, arm, directory, stats):
    """Bind the official checkpoint and training context and return physical-unit forecasts."""
    import torch

    path = weights(task)
    if not torch.cuda.is_available():
        raise RuntimeError("The pretrained-model study requires CUDA")
    train = task.sample(arm, "train")
    x = stats.inputs(train["X"]).reshape(len(train["y"]), -1)
    y = stats.labels(train["y"])
    torch.set_num_threads(2)
    torch.manual_seed(task.seed)
    np.random.seed(task.seed)
    jobs = 4 if task.experiment in ("reference", "joint", "raw-selection") else 2
    if task.model == "TabPFN":
        from tabpfn import TabPFNRegressor

        # The pinned backend maps True directly to the registered native factor-eight branch.
        model = TabPFNRegressor(
            n_estimators=8,
            auto_scale_n_estimators=False,
            model_path=path,
            device="cuda:0",
            random_state=task.seed,
            fit_mode="fit_preprocessors",
            show_progress_bar=False,
            categorical_features_indices=[],
            inference_precision="auto",
            memory_saving_mode=True,
            ignore_pretraining_limits=False,
            n_preprocessing_jobs=jobs,
        )
        model.fit(x, y)
        cpu_state = torch.random.get_rng_state().clone()
        gpu = torch.cuda.current_device()
        gpu_state = torch.cuda.get_rng_state(gpu).clone()

        def predict_native(features):
            with torch.random.fork_rng(devices=[gpu]):
                torch.random.set_rng_state(cpu_state)
                torch.cuda.set_rng_state(gpu_state, gpu)
                return model.predict(features)
    else:
        from tabicl import TabICLRegressor

        policies = {
            k: dict(
                offload="cpu",
                safety_factor=0.2,
                min_batch_size=1,
                use_amp=False,
                use_fa3=False,
            )
            for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")
        }
        model = TabICLRegressor(
            model_path=path,
            checkpoint_version=path.name,
            allow_auto_download=False,
            device="cuda:0",
            random_state=task.seed,
            n_estimators=8,
            batch_size=8,
            kv_cache=False,
            use_amp=False,
            use_fa3=False,
            offload_mode="cpu",
            inference_config=policies,
            disk_offload_dir=str(task.work / "offload" / task.case / task.model / arm),
            n_jobs=min(jobs, os.cpu_count()) - os.cpu_count() - 1,
        )
        model.fit(x, y)
        predict_native = model.predict

    def predict(row):
        features = stats.inputs(row["X"]).reshape(len(row["y"]), -1)
        return stats.restore(predict_native(features))

    return predict
