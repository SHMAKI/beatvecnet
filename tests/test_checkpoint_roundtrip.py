from __future__ import annotations

import pytest
import torch

from scripts.test import extract_standardization


def test_checkpoint_standardization_round_trip():
    ckpt = {
        "sample_training_channel_mean": [1.0, 2.0],
        "sample_training_channel_std": [3.0, 4.0],
    }
    assert extract_standardization(ckpt) == ([1.0, 2.0], [3.0, 4.0])


def test_checkpoint_missing_stats_errors():
    with pytest.raises(KeyError):
        extract_standardization({"model_state_dict": {}})
