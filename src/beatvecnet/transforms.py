from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ChannelStandardization:
    mean: Sequence[float]
    std: Sequence[float]
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        if len(self.mean) != len(self.std):
            raise ValueError("mean and std must have the same length")
        if any(float(s) <= 0 for s in self.std):
            raise ValueError("all standard deviations must be positive")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
        return (x - mean) / (std + self.eps)


class RandomTemporalShift(torch.nn.Module):
    def __init__(
        self,
        maximum_shift_fraction: float = 0.05,
        application_probability: float = 0.5,
        boundary_fill: str = "zero",
    ):
        super().__init__()
        self.maximum_shift_fraction = float(maximum_shift_fraction)
        self.application_probability = float(application_probability)
        if boundary_fill != "zero":
            raise ValueError("only zero boundary_fill is supported")
        self.boundary_fill = boundary_fill

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.maximum_shift_fraction <= 0 or self.application_probability <= 0:
            return x
        if torch.rand(()) > self.application_probability:
            return x
        t = x.shape[-1]
        max_shift = int(round(t * self.maximum_shift_fraction))
        if max_shift <= 0:
            return x
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        if shift == 0:
            return x
        out = torch.zeros_like(x)
        if shift > 0:
            out[..., shift:] = x[..., :-shift]
        else:
            s = -shift
            out[..., :-s] = x[..., s:]
        return out


class BeatTransform:
    def __init__(
        self,
        standardization: ChannelStandardization | None = None,
        temporal_shift: RandomTemporalShift | None = None,
        train: bool = False,
    ):
        self.standardization = standardization
        self.temporal_shift = temporal_shift
        self.train = bool(train)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.standardization is not None:
            x = self.standardization(x)
        if self.train and self.temporal_shift is not None:
            x = self.temporal_shift(x)
        return x
