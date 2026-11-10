"""Synchronized six-node local graph and deterministic masked-neighbor pooling control."""

import json

import numpy as np
import torch

from wdaoh.models.graph import LocalSAGETCN, tensors
from wdaoh.models.sequence import device
from wdaoh.models.training import Scale
from wdaoh.protocol import LMH, write_json


def fit(task, arm, train, directory):
    """Use the three frozen checkpoint epochs and equal-stratum 2024 MAE to choose one set of weights."""
    cfg = task.config(arm)
    dev = device(task.seed)
    model = LocalSAGETCN(arm, cfg["width"]).to(dev)
    with np.load(task.inputs / "graph/topology.npz") as topology:
        top = {"adjacency": topology["adjacency"]}
    x, adj, normal = tensors(train, top)
    write_json(directory / "graph_scale.json", normal)
    xx, aa, yy = (
        torch.as_tensor(v, device=dev)
        for v in (x, adj, (train["y"] - normal["y_mean"]) / normal["y_std"])
    )
    validation = {}
    for g in LMH:
        q = task.sample(arm, "validation_" + g)
        vx, va, _ = tensors(q, top, normal)
        validation[g] = (
            torch.as_tensor(vx, device=dev),
            torch.as_tensor(va, device=dev),
            q["y"],
        )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    best = float("inf")
    selected = 0
    scores = {}
    for epoch in range(cfg["epochs"]):
        model.train()
        order = np.random.default_rng(task.seed + epoch * 37).permutation(len(yy))
        for k in range(0, len(yy), cfg["batch_size"]):
            idx = order[k : k + cfg["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model(xx[idx], aa[idx]), yy[idx])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite graph loss")
            loss.backward()
            optimizer.step()
        if epoch + 1 in cfg["checkpoint_epochs"]:
            model.eval()
            group_errors = []
            for vx, va, targets in validation.values():
                raw = (
                    batch_predict(model, vx, va, cfg["batch_size"]) * normal["y_std"]
                    + normal["y_mean"]
                )
                group_errors.append(float(np.abs(raw - targets).mean()))
            score = float(np.mean(group_errors))
            scores[epoch + 1] = score
            if score < best:
                best, selected = score, epoch + 1
                torch.save(model.state_dict(), directory / "model.pt")
    return dict(checkpoint_epoch=selected, validation_checkpoint_scores=scores), Scale(
        np.empty(0), np.empty(0), normal["y_mean"], normal["y_std"], False
    )


def batch_predict(model, x, adj, batch):
    """Evaluate a fixed graph checkpoint in batches with gradient recording disabled."""
    with torch.no_grad():
        return (
            torch.cat(
                [
                    model(x[k : k + batch], adj[k : k + batch])
                    for k in range(0, len(x), batch)
                ]
            )
            .cpu()
            .numpy()
        )


def load(task, arm, directory, stats):
    """Restore the chosen validation checkpoint and its training-only graph and target scales."""
    cfg = task.config(arm)
    dev = device(task.seed)
    model = LocalSAGETCN(arm, cfg["width"]).to(dev)
    model.load_state_dict(
        torch.load(directory / "model.pt", weights_only=True, map_location=dev)
    )
    model.eval()
    normal = json.loads((directory / "graph_scale.json").read_text())
    with np.load(task.inputs / "graph/topology.npz") as topology:
        top = {"adjacency": topology["adjacency"]}

    def predict(row):
        x, adj, _ = tensors(row, top, normal)
        pred = batch_predict(
            model,
            torch.as_tensor(x, device=dev),
            torch.as_tensor(adj, device=dev),
            cfg["batch_size"],
        )
        return stats.restore(pred)

    return predict
