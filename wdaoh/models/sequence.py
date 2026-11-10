"""Native endpoints and deterministic chronological fitting for the study's sequence predictors."""

import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from wdaoh.models.grud import GRUD
from wdaoh.models.packing import indices, packed
from wdaoh.models.training import Scale, chronological_split, scale
from wdaoh.models.upstream import architecture


class Endpoint(nn.Module):
    """Predict the final horizon and target channel, with the registered DLinear side-information residual."""

    def __init__(self, native, name, channels):
        super().__init__()
        self.native = native
        self.name = name
        self.side = (
            nn.Linear(24 * (channels - 1), 1, bias=False)
            if name == "DLinear+side" and channels > 1
            else None
        )
        if self.side is not None:
            nn.init.zeros_(self.side.weight)

    def forward(self, x):
        y = (
            self.native(x, None, None, None)
            if self.name == "iTransformer"
            else self.native(x)
        )
        endpoint = y[:, -1, -1]
        if self.side is not None:
            endpoint = endpoint + self.side(x[:, :, :-1].reshape(len(x), -1)).squeeze(
                -1
            )
        return endpoint


def endpoint(work, model, horizon, channels, config, stats, lookback=24):
    """Instantiate frozen upstream architecture settings for a 24- or 168-hour input sequence."""
    dim = config["d_model"]
    if model == "GRU-D":
        return GRUD(channels, dim, (stats.mean, stats.deviation))
    options = SimpleNamespace(
        seq_len=24,
        pred_len=horizon,
        enc_in=channels,
        individual=False,
        output_attention=False,
        use_norm=False,
        class_strategy="projection",
        embed="timeF",
        freq="h",
        dropout=0.1,
        factor=1,
        n_heads=4,
        d_model=dim,
        d_ff=4 * dim,
        e_layers=2,
        activation="gelu",
        task_name="long_term_forecast",
        features="MS",
        patch_len=4,
        stem_ratio=2,
        downsample_ratio=2,
        ffn_ratio=2,
        num_blocks=[2],
        large_size=[config["kernel"]],
        small_size=[3],
        dims=[dim] * 4,
        dw_dims=[dim] * 4,
        small_kernel_merged=False,
        head_dropout=0.1,
        use_multi_scale=False,
        revin=False,
        affine=False,
        subtract_last=False,
        kernel_size=25,
        patch_size=4,
        patch_stride=2,
        decomposition=False,
    )
    native = architecture(work, model)
    result = Endpoint(native(options), model, channels)
    if lookback != 24:
        # The frozen long-history worker consumes the initial model's RNG draws before constructing 168 hours.
        options.seq_len = lookback
        result = Endpoint(native(options), model, channels)
    return result


def device(seed):
    """Use the original CUDA execution contract for scientific training."""
    if os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
        raise ValueError("The frozen CUDA fit uses CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Sequence training requires the CUDA runtime and a compatible GPU"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda")


def input_tensor(task, arm, x, stats):
    """Preserve native sequences or apply the original lossless flat-column packing for DLinear."""
    if x.ndim == 3:
        values = stats.inputs(x)
    elif task.provider == "EEA":
        values = x.reshape(len(x), 24, -1)
    else:
        columns = settings_for_arm(task, arm)
        values = packed(
            ((x - stats.mean) / stats.deviation).astype("float32"), *indices(columns)
        )
    return torch.as_tensor(values, device="cuda")


def settings_for_arm(task, arm):
    return json.loads((task.inputs / arm / "columns.json").read_text())


def seq_scale(task, x, y):
    """Fit raw channels or engineered columns using the scaling convention of this study arm."""
    if x.ndim == 2 and task.provider != "EEA":
        deviation = x.std(0)
        return Scale(
            x.mean(0),
            np.where(deviation < 1e-6, 1, deviation),
            float(y.mean()),
            max(float(y.std()), 1e-6),
            False,
        )
    return scale(x, y, x.ndim == 3, 1e-8 if task.provider == "EEA" else 1e-6)


def predict_tensor(model, x, batch):
    """Evaluate a sequence model in fixed-order batches without gradients."""
    with torch.no_grad():
        return torch.cat([model(x[i : i + batch]) for i in range(0, len(x), batch)])


def fit(task, arm, train, directory):
    """Use Adam and MAE for the purged stopping fit, then reseed and refit on the entire training budget."""
    cfg = task.config(arm)
    fit_idx, stop_idx, boundary = chronological_split(train)
    x, y = train["X"], train["y"]
    batch = cfg["batch_size"]

    def initialize(ids):
        dev = device(task.seed)
        st = seq_scale(task, x[ids], y[ids])
        xx = input_tensor(task, arm, x[ids], st)
        yy = torch.as_tensor(st.labels(y[ids]), device=dev)
        m = endpoint(
            task.work, task.model, task.horizon, xx.shape[-1], cfg, st, xx.shape[1]
        ).to(dev)
        optimizer = torch.optim.Adam(
            m.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
        return st, xx, yy, m, optimizer

    def epoch_step(model, optimizer, xx, yy, order):
        model.train()
        for start in range(0, len(order), batch):
            ids = order[start : start + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.l1_loss(model(xx[ids]), yy[ids])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()

    st, xx, yy, model, optimizer = initialize(fit_idx)
    vx = input_tensor(task, arm, x[stop_idx], st)
    vy = torch.as_tensor(st.labels(y[stop_idx]), device="cuda")
    best_epoch, best_loss, stall = 0, float("inf"), 0
    for epoch in range(cfg["max_epochs"]):
        order = np.random.default_rng(task.seed + epoch * 37).permutation(len(fit_idx))
        epoch_step(model, optimizer, xx, yy, order)
        model.eval()
        loss = float(nn.functional.l1_loss(predict_tensor(model, vx, batch), vy))
        if loss < best_loss:
            best_epoch, best_loss, stall = epoch + 1, loss, 0
        else:
            stall += 1
        if stall >= cfg["patience"]:
            break
    if not best_epoch:
        raise ValueError("No finite inner stopping score")
    del optimizer, model, xx, yy, vx, vy
    st, xx, yy, model, optimizer = initialize(np.arange(len(y)))
    for epoch in range(best_epoch):
        order = np.random.default_rng(task.seed + epoch * 37 + 10000).permutation(
            len(y)
        )
        epoch_step(model, optimizer, xx, yy, order)
    torch.save(
        dict(model=model.state_dict(), channels=xx.shape[-1], lookback=xx.shape[1]),
        directory / "model.pt",
    )
    return dict(
        iterations=best_epoch,
        boundary=boundary,
        fit_rows=len(fit_idx),
        stop_rows=len(stop_idx),
    ), st


def load(task, arm, directory, stats):
    """Restore fitted weights and reconstruct the same sequence representation for inference."""
    dev = device(task.seed)
    state = torch.load(directory / "model.pt", weights_only=True, map_location=dev)
    cfg = task.config(arm)
    model = endpoint(
        task.work,
        task.model,
        task.horizon,
        state["channels"],
        cfg,
        stats,
        state["lookback"],
    ).to(dev)
    model.load_state_dict(state["model"])
    model.eval()

    def predict(query):
        x = input_tensor(task, arm, query["X"], stats)
        return stats.restore(predict_tensor(model, x, cfg["batch_size"]).cpu().numpy())

    return predict
