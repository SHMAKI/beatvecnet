from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetCfg:
    n_classes: int
    size_y: int = 500
    size_t: int = 150
    in_ch: Optional[int] = 8
    mmap_mode: Optional[str] = "r"
    dtype: torch.dtype = torch.float32
    metadata_columns: Optional[Dict[str, str]] = None


DEFAULT_METADATA_ALIASES: dict[str, tuple[str, ...]] = {
    "task_name": ("task_name", "task", "batch"),
    "well_id": ("well_id", "Well", "WELL", "well"),
    "roi_id": ("roi_id", "ROI", "roi"),
    "beat_id": ("beat_id", "BEAT", "beat"),
    "cond_raw": ("cond_raw", "condition", "cond"),
    "sample_id": ("sample_id", "sample"),
    "class_id": ("class_id", "CLASS_ID", "ANS"),
}


class SampleDataset(Dataset):
    """
    Public-facing dataset for BeatVecNet-paper.

    Expected manifest columns:
      - path      : relative or absolute path to .npy
      - class_id  : integer label
      - split     : train / val / test   (optional if filtered outside)
      - sample_id : optional
      - class     : optional
      - cond_raw  : optional
      - well_id   : optional

    Returns:
      - x: torch.FloatTensor, shape (C, H, T)
      - y: torch.LongTensor, scalar
      - meta: optional dict
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: DatasetCfg,
        repo_root: str | Path = ".",
        transform=None,
        path_col: str = "path",
        label_col: str = "class_id",
        return_meta: bool = True,
        enforce_shape: bool = True,
        metadata_columns: Optional[Dict[str, str]] = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.cfg = cfg
        self.repo_root = Path(repo_root)
        self.transform = transform
        self.path_col = path_col
        self.label_col = label_col
        self.return_meta = return_meta
        self.enforce_shape = enforce_shape
        self.metadata_columns = metadata_columns or cfg.metadata_columns or {}

        for c in [self.path_col, self.label_col]:
            if c not in self.df.columns:
                raise KeyError(f"required column not found: {c}")

        self.df[self.label_col] = (
            pd.to_numeric(self.df[self.label_col], errors="coerce")
            .fillna(-1)
            .astype(int)
        )
        self._metadata_map = self._resolve_metadata_map()

    def _resolve_metadata_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for canonical, aliases in DEFAULT_METADATA_ALIASES.items():
            configured = self.metadata_columns.get(canonical)
            if configured:
                if configured not in self.df.columns:
                    raise KeyError(
                        f"metadata column mapping for {canonical!r} points to missing column {configured!r}"
                    )
                mapping[canonical] = configured
                continue
            for col in aliases:
                if col in self.df.columns:
                    mapping[canonical] = col
                    break
        return mapping

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, p: str) -> Path:
        p = Path(str(p))
        if p.is_absolute():
            return p
        return self.repo_root / p

    def _to_cht(self, x: np.ndarray) -> np.ndarray:
        """
        Convert input array to (C, H, T).

        Supported:
          - (C, H, T)
          - (H, T, C)
        """
        if x.ndim != 3:
            raise ValueError(f"expected 3D array, got shape={x.shape}")

        in_ch = self.cfg.in_ch

        if in_ch is not None:
            if x.shape[0] == in_ch:
                return x
            if x.shape[-1] == in_ch:
                return np.transpose(x, (2, 0, 1))

        # fallback heuristic: smallest axis is channel
        if x.shape[0] <= x.shape[-1]:
            return x
        return np.transpose(x, (2, 0, 1))

    def _check_shape(self, x: np.ndarray) -> None:
        c, h, t = x.shape
        if (h, t) != (self.cfg.size_y, self.cfg.size_t):
            msg = (
                f"shape mismatch: got (C,H,T)=({c},{h},{t}), "
                f"expected H,T=({self.cfg.size_y},{self.cfg.size_t})"
            )
            if self.enforce_shape:
                raise ValueError(msg)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        y = int(row[self.label_col])
        path = self._resolve_path(row[self.path_col])

        if not path.exists():
            raise FileNotFoundError(f"npy not found: {path}")

        x = np.load(path, mmap_mode=self.cfg.mmap_mode) if self.cfg.mmap_mode else np.load(path)
        x = np.asarray(x)
        x = self._to_cht(x)
        x = np.asarray(x, dtype=np.float32)
        x = np.ascontiguousarray(x).copy()   # writable な独立配列にする
        self._check_shape(x)

        xt = torch.from_numpy(x)
        if self.cfg.dtype != torch.float32:
            xt = xt.to(dtype=self.cfg.dtype)

        if self.transform is not None:
            try:
                xt = self.transform(xt)
            except TypeError:
                xt = self.transform(xt)

        yt = torch.tensor(y, dtype=torch.long)

        if not self.return_meta:
            return xt, yt

        meta: Dict[str, Any] = {
            "index": idx,
            "path": str(path),
            "split": str(row.get("split", "")),
            "class": str(row.get("class", "")),
            "class_id": int(y),
        }
        for canonical, source in self._metadata_map.items():
            value = row.get(source, "")
            if pd.isna(value):
                value = ""
            if canonical == "class_id":
                meta[canonical] = int(y)
            else:
                meta[canonical] = value
        return xt, yt, meta


def load_manifest(
    manifest_path: str | Path,
    split: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    if split is not None:
        df = df[df["split"].astype(str).str.lower() == str(split).lower()].copy()

    return df.reset_index(drop=True)


def normalize_manifest_metadata_columns(
    df: pd.DataFrame,
    metadata_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    out = df.copy()
    mapping = metadata_columns or {}
    for canonical, aliases in DEFAULT_METADATA_ALIASES.items():
        if canonical in out.columns:
            continue
        source = mapping.get(canonical)
        if source and source in out.columns:
            out[canonical] = out[source]
            continue
        for col in aliases:
            if col in out.columns:
                out[canonical] = out[col]
                break
    return out
