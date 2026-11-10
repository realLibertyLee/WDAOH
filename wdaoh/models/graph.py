"""Explicit custom first-order, missing-aware GraphSAGE-style + two TCN blocks.

This is an adapted local model, not a reproduction of GWN or D2STGNN.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def masked_aggregate(h, valid, adj):
    # adj B,receiver,sender; valid B,T,sender. Invalid senders never send zeros
    # as measurements: they receive exactly zero weight before renormalization.
    """Aggregate only observed senders and return an availability flag for each receiver and hour."""
    weights = adj[:, None, :, :] * valid[:, :, None, :]
    denom = weights.sum(-1, keepdim=True)
    weights = weights / denom.clamp_min(1e-12)
    return torch.einsum("btij,btjd->btid", weights, h), denom.squeeze(-1) > 0


class LocalSAGETCN(nn.Module):
    """Predict at the target node using the registered six-node information arm and causal convolutions."""

    def __init__(self, arm, width=16):
        super().__init__()
        assert arm in ["SELF_ONLY", "GRAPH_NEIGHBORS", "MASKED_NEIGHBOR_POOL"]
        self.arm = arm
        self.encoder = nn.Linear(10, width)
        self.self_map = nn.Linear(width, width)
        self.message_map = nn.Linear(width, width, bias=False)
        self.tcn1 = nn.Conv1d(width, width, 3, dilation=1)
        self.tcn2 = nn.Conv1d(width, width, 3, dilation=2)
        self.head = nn.Linear(24 * width, 1)

    def forward(self, x, adj):
        valid = x[:, :, :, 1] > 0.5
        # Enforce sanitization also at model boundary, including malicious masked NaN.
        x = torch.cat(
            [
                torch.where(valid, x[:, :, :, 0], torch.zeros_like(x[:, :, :, 0]))[
                    :, :, :, None
                ],
                x[:, :, :, 1:],
            ],
            dim=-1,
        )
        if self.arm == "SELF_ONLY":
            adj = torch.eye(6, device=x.device, dtype=x.dtype)[None].expand(
                len(x), -1, -1
            )
        h = F.relu(self.encoder(x))
        if self.arm == "MASKED_NEIGHBOR_POOL":
            raw, available = masked_aggregate(x, valid, adj)
            message = F.relu(self.encoder(raw)) * available[:, :, :, None]
        else:
            message, _ = masked_aggregate(h, valid, adj)
        h = F.relu(self.self_map(h) + self.message_map(message))[:, :, 0, :].transpose(
            1, 2
        )
        h = F.relu(h + self.tcn1(F.pad(h, (2, 0))))
        h = F.relu(h + self.tcn2(F.pad(h, (4, 0))))
        return self.head(h.flatten(1)).squeeze(-1)


def tensors(z, top, st=None):
    """Six synchronized nodes, common calendar6 + target geo2 at every node.

    Node coordinates remain audit/adjacency metadata. Broadcasting target geo keeps
    explicit covariates identical to the existing native neighbor ablation.
    """
    x = z["X"]
    n = len(x)
    assert x.shape == (n, 24, 30)
    v = np.stack([x[:, :, 29]] + [x[:, :, i] for i in range(19, 29, 2)], axis=2)
    m = np.stack([x[:, :, 8]] + [x[:, :, i] for i in range(20, 29, 2)], axis=2)
    assert np.array_equal(m, np.isfinite(v).astype("float32"))
    cov = np.broadcast_to(x[:, :, :8, None].transpose(0, 1, 3, 2), (n, 24, 6, 8)).copy()
    if st is None:
        observed = v[:, :, 0][m[:, :, 0] > 0.5]
        assert len(observed) > 0
        st = dict(
            value_mean=float(observed.mean()),
            value_std=max(float(observed.std()), 1e-6),
            cov_mean=cov[:, :, 0].mean((0, 1)).tolist(),
            cov_std=np.maximum(cov[:, :, 0].std((0, 1)), 1e-6).tolist(),
            y_mean=float(z["y"].mean()),
            y_std=max(float(z["y"].std()), 1e-6),
        )
    value = np.where(
        m > 0.5,
        (np.nan_to_num(v, nan=st["value_mean"]) - st["value_mean"]) / st["value_std"],
        0,
    )
    cov = (cov - np.asarray(st["cov_mean"])) / np.asarray(st["cov_std"])
    features = np.concatenate(
        [value[:, :, :, None], m[:, :, :, None], cov], axis=-1
    ).astype("float32")
    assert np.isfinite(features).all()
    adj = top["adjacency"][z["site_index"]].astype("float32")
    return features, adj, st
