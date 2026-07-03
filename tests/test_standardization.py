from __future__ import annotations

import numpy as np
import pandas as pd

from beatvecnet.dataset import DatasetCfg
from beatvecnet.standardization import compute_channel_mean_std


def test_channel_mean_std_uses_training_rows_only(tmp_path):
    paths = []
    train_arrays = [
        np.full((2, 2, 2), 1.0, dtype=np.float32),
        np.full((2, 2, 2), 3.0, dtype=np.float32),
    ]
    test_array = np.full((2, 2, 2), 100.0, dtype=np.float32)
    for i, arr in enumerate([*train_arrays, test_array]):
        p = tmp_path / f"s{i}.npy"
        np.save(p, arr)
        paths.append(p)
    df = pd.DataFrame(
        {
            "path": [str(p) for p in paths],
            "class_id": [0, 0, 0],
            "split": ["train", "train", "test"],
        }
    )
    cfg = DatasetCfg(n_classes=1, size_y=2, size_t=2, in_ch=2, mmap_mode=None)
    mean, std = compute_channel_mean_std(df[df["split"] == "train"], cfg=cfg)
    assert np.allclose(mean, [2.0, 2.0])
    assert np.allclose(std, [1.0, 1.0])
