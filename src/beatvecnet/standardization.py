from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from beatvecnet.dataset import DatasetCfg, SampleDataset


def compute_channel_mean_std(
    df_train: pd.DataFrame,
    cfg: DatasetCfg,
    repo_root: str | Path = ".",
    path_col: str = "path",
) -> tuple[np.ndarray, np.ndarray]:
    n_total = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    ds = SampleDataset(
        df=df_train,
        cfg=cfg,
        repo_root=repo_root,
        path_col=path_col,
        return_meta=False,
        transform=None,
        enforce_shape=True,
    )

    for i in range(len(ds)):
        x, _ = ds[i]
        arr = x.numpy().astype(np.float64, copy=False)
        c = arr.shape[0]
        flat = arr.reshape(c, -1)
        if not np.isfinite(flat).all():
            raise ValueError(f"non-finite tensor values while fitting standardization at row {i}")
        batch_n = flat.shape[1]
        batch_mean = flat.mean(axis=1)
        batch_var = flat.var(axis=1)
        batch_m2 = batch_var * batch_n

        if mean is None:
            mean = batch_mean
            m2 = batch_m2
            n_total = batch_n
            continue

        assert m2 is not None
        delta = batch_mean - mean
        new_total = n_total + batch_n
        mean = mean + delta * (batch_n / new_total)
        m2 = m2 + batch_m2 + (delta * delta) * (n_total * batch_n / new_total)
        n_total = new_total

    if mean is None or m2 is None or n_total == 0:
        raise ValueError("cannot fit channel standardization from an empty training split")

    std = np.sqrt(m2 / n_total)
    if np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError("invalid fitted channel standard deviations")
    return mean.astype(np.float32), std.astype(np.float32)


def save_standardization_json(path: str | Path, mean: np.ndarray, std: np.ndarray) -> None:
    payload = {
        "scope": "training_split_only",
        "variance_definition": "population",
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
