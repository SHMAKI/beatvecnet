from __future__ import annotations

import torch

from beatvecnet.transforms import RandomTemporalShift


def test_temporal_shift_p_zero_is_identity():
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    y = RandomTemporalShift(0.5, application_probability=0.0)(x)
    assert torch.equal(x, y)


def test_temporal_shift_zero_fills_without_wrap(monkeypatch):
    x = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5)

    def fake_rand(*args, **kwargs):
        return torch.tensor(0.0)

    def fake_randint(*args, **kwargs):
        return torch.tensor([2])

    monkeypatch.setattr(torch, "rand", fake_rand)
    monkeypatch.setattr(torch, "randint", fake_randint)
    y = RandomTemporalShift(0.5, application_probability=1.0)(x)
    assert y.shape == x.shape
    assert torch.equal(y, torch.tensor([[[0.0, 0.0, 0.0, 1.0, 2.0]]]))
