from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def day0_beat_scale(
    tensor: np.ndarray,
    vx_channels: Sequence[int] = (0, 1, 2, 3),
    vy_channels: Sequence[int] = (4, 5, 6, 7),
    top_fraction: float = 0.05,
) -> float:
    x = np.asarray(tensor, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"expected tensor shape (C,H,T), got {x.shape}")
    if len(vx_channels) != len(vy_channels):
        raise ValueError("vx_channels and vy_channels must have the same length")
    mags = []
    for vx, vy in zip(vx_channels, vy_channels):
        mag = np.sqrt(np.square(x[int(vx)]) + np.square(x[int(vy)]))
        mags.append(mag.reshape(-1))
    values = np.concatenate(mags)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("no positive finite motion magnitudes for day0 beat scale")
    n_top = max(1, int(np.ceil(values.size * float(top_fraction))))
    top = np.partition(values, values.size - n_top)[-n_top:]
    scale = float(np.median(top))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid day0 beat scale")
    return scale


def fit_day0_well_scales(
    manifest: pd.DataFrame,
    repo_root: str | Path,
    path_col: str = "path",
    well_col: str = "well_id",
    day_col: str = "day",
    day0_value: str | int | float = 0,
) -> pd.DataFrame:
    if well_col not in manifest.columns or day_col not in manifest.columns:
        raise KeyError(f"manifest must contain {well_col!r} and {day_col!r}")
    root = Path(repo_root)
    rows = []
    day0 = manifest[manifest[day_col].astype(str) == str(day0_value)]
    for well, g in day0.groupby(well_col):
        beat_scales = []
        for _, row in g.iterrows():
            path = Path(str(row[path_col]))
            if not path.is_absolute():
                path = root / path
            scale = day0_beat_scale(np.load(path))
            beat_scales.append(scale)
        if not beat_scales:
            continue
        well_scale = float(np.median(np.asarray(beat_scales, dtype=float)))
        if not np.isfinite(well_scale) or well_scale <= 0:
            raise ValueError(f"invalid day0 well scale for {well!r}")
        rows.append({"well_id": well, "day0_scale": well_scale, "n_day0_beats": len(beat_scales)})
    if not rows:
        raise ValueError("no day0 rows available to fit well scales")
    return pd.DataFrame(rows)


def normalize_manifest_tensors(
    manifest: pd.DataFrame,
    scales: pd.DataFrame,
    repo_root: str | Path,
    output_dir: str | Path,
    path_col: str = "path",
    well_col: str = "well_id",
) -> pd.DataFrame:
    root = Path(repo_root)
    out_dir = Path(output_dir)
    tensor_dir = out_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    scale_map = dict(zip(scales["well_id"].astype(str), scales["day0_scale"].astype(float)))
    out_manifest = manifest.copy()
    new_paths = []
    exclusion_reasons = []
    for i, row in manifest.iterrows():
        well = str(row[well_col])
        scale = scale_map.get(well)
        if scale is None or not np.isfinite(scale) or scale <= 0:
            new_paths.append("")
            exclusion_reasons.append("missing_or_invalid_day0_scale")
            continue
        in_path = Path(str(row[path_col]))
        if not in_path.is_absolute():
            in_path = root / in_path
        x = np.load(in_path).astype(np.float32, copy=False)
        out_path = tensor_dir / f"{Path(str(row[path_col])).stem}.npy"
        np.save(out_path, x / float(scale))
        try:
            rel = out_path.relative_to(root)
            new_paths.append(str(rel))
        except ValueError:
            new_paths.append(str(out_path))
        exclusion_reasons.append("")
    out_manifest[path_col] = new_paths
    out_manifest["day0_normalization_exclusion_reason"] = exclusion_reasons
    return out_manifest
