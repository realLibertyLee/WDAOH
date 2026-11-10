"""Mask-aware recurrent endpoint for the ten- and thirty-channel EPA inputs."""

import torch
from torch import nn


def layout(channels):
    """Return paired value and observation-mask columns for the native 10- or 30-channel schema."""
    return (
        ([9], [8])
        if channels == 10
        else (list(range(9, 29, 2)) + [29], list(range(10, 30, 2)) + [8])
    )


class InputMasks(nn.Module):
    """Recover binary observation indicators from standardized native channels."""

    def __init__(self, channels, stats):
        super().__init__()
        self.values, self.masks = layout(channels)
        self.channels = channels
        self.register_buffer("input_mean", torch.as_tensor(stats[0]))
        self.register_buffer("input_std", torch.as_tensor(stats[1]))

    def decoded(self, x):
        return x * self.input_std + self.input_mean

    def observed(self, x):
        return (self.decoded(x)[..., self.masks] > 0.5).to(x.dtype)


class GRUD(InputMasks):
    """Decay missing measurements and the hidden state before each adapted GRU update."""

    def __init__(self, channels, d, stats):
        super().__init__(channels, stats)
        v = len(self.values)
        self.hidden = d
        self.gamma_x_w = nn.Parameter(torch.ones(v))
        self.gamma_x_b = nn.Parameter(torch.zeros(v))
        self.gamma_h = nn.Linear(v, d)
        self.cell = nn.GRUCell(v * 2 + 8, d)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d, 1)
        )

    def forward(self, x):
        mask = self.observed(x)
        value = x[..., self.values]
        b = len(x)
        h = x.new_zeros(b, self.hidden)
        last = x.new_zeros(b, len(self.values))
        delta = x.new_ones(b, len(self.values))
        for t in range(24):
            if t:
                delta = 1 + (1 - mask[:, t - 1]) * delta
            gx = torch.exp(-torch.relu(delta * self.gamma_x_w + self.gamma_x_b))
            h = h * torch.exp(-torch.relu(self.gamma_h(delta)))
            m = mask[:, t]
            filled = m * value[:, t] + (1 - m) * gx * last
            h = self.cell(torch.cat([filled, m, x[:, t, :8]], -1), h)
            last = m * value[:, t] + (1 - m) * last
        return self.head(h).squeeze(-1)
