from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


def make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """
    Create GroupNorm with a valid number of groups dividing num_channels.
    """
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return nn.GroupNorm(g, num_channels)


@dataclass
class ModelCfg:
    n_output: int = 4
    n_layers: int = 4
    kernel_size: int = 3
    mid_units: int = 64
    dropout_rate: float = 0.3
    n_filters: Sequence[int] = (32, 16, 16, 32)
    n_channel_in: int = 8


class BeatVecNet(nn.Module):
    """
    Public-facing minimal BeatVecNet implementation.

    Input:
        x: (B, C, H, T), typically (B, 8, 500, 150)

    Channel convention:
        x[:, :4, :, :] -> x-branch
        x[:, 4:, :, :] -> y-branch
    """

    def __init__(self, cfg: ModelCfg):
        super().__init__()

        if cfg.n_channel_in != 8:
            raise ValueError(f"Expected n_channel_in=8, got {cfg.n_channel_in}")
        if len(cfg.n_filters) != cfg.n_layers:
            raise ValueError(
                f"len(n_filters) must equal n_layers: "
                f"{len(cfg.n_filters)} != {cfg.n_layers}"
            )

        self.cfg = cfg
        self.activation = nn.ELU(inplace=False)
        self.flatten = nn.Flatten()

        self.branch_x_layers = nn.ModuleList()
        self.branch_y_layers = nn.ModuleList()

        current_channels = cfg.n_channel_in // 2  # 4 channels per branch

        def conv_block(cin: int, cout: int):
            return [
                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=cfg.kernel_size,
                    padding="same",
                    bias=False,
                ),
                make_group_norm(cout, 8),
                nn.ELU(inplace=False),
                nn.MaxPool2d(2),
            ]

        # branch blocks
        for i in range(cfg.n_layers):
            cout = int(cfg.n_filters[i])
            block = conv_block(current_channels, cout)

            self.branch_x_layers.extend(block)
            self.branch_y_layers.extend([copy.deepcopy(layer) for layer in block])

            current_channels = cout

        # fusion
        fuse_in = int(cfg.n_filters[-1]) * 2
        fuse_out = int(cfg.n_filters[-1])

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(
                fuse_in,
                fuse_out,
                kernel_size=cfg.kernel_size,
                padding="same",
                bias=False,
            ),
            make_group_norm(fuse_out, 8),
            nn.ELU(inplace=False),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(fuse_out, cfg.mid_units),
            nn.ELU(inplace=False),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.mid_units, cfg.n_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B,C,H,T), got {tuple(x.shape)}")
        if x.shape[1] != self.cfg.n_channel_in:
            raise ValueError(
                f"Expected {self.cfg.n_channel_in} input channels, got {x.shape[1]}"
            )

        x_x = x[:, :4, :, :]
        x_y = x[:, 4:, :, :]

        for layer in self.branch_x_layers:
            x_x = layer(x_x)

        for layer in self.branch_y_layers:
            x_y = layer(x_y)

        fused = torch.cat([x_x, x_y], dim=1)
        fused = self.fuse_conv(fused)
        fused = self.adaptive_pool(fused)
        fused = self.flatten(fused)
        logits = self.classifier(fused)

        return logits