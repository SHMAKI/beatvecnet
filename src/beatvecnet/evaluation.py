from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


DEFAULT_LABELS = [0, 1, 2, 3]


def probability_columns(n_classes: int, prefix: str = "prob_") -> list[str]:
    return [f"{prefix}{i}" for i in range(n_classes)]


def has_hierarchy_metadata(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in ["task_name", "well_id", "roi_id"])


def _first_consistent(s: pd.Series, group_name: str) -> int:
    vals = pd.Series(s).dropna().astype(int).unique()
    if len(vals) != 1:
        raise ValueError(f"true class is not consistent within {group_name}: {vals.tolist()}")
    return int(vals[0])


def _first_optional(s: pd.Series):
    vals = pd.Series(s).dropna()
    return vals.iloc[0] if len(vals) else ""


def validate_prediction_table(
    df: pd.DataFrame,
    n_classes: int,
    prob_cols: Sequence[str] | None = None,
    true_col: str = "y_true",
) -> list[str]:
    prob_cols = list(prob_cols or probability_columns(n_classes))
    required = ["task_name", "well_id", "roi_id", true_col, *prob_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"hierarchical evaluation requires columns: {missing}")
    if len(prob_cols) != n_classes:
        raise ValueError(f"expected {n_classes} probability columns, got {len(prob_cols)}")
    probs = df[prob_cols].to_numpy(dtype=float)
    if not np.isfinite(probs).all():
        raise ValueError("probability columns contain non-finite values")
    df.groupby(["task_name", "well_id", "roi_id"])[true_col].apply(
        lambda s: _first_consistent(s, "ROI")
    )
    df.groupby(["task_name", "well_id"])[true_col].apply(
        lambda s: _first_consistent(s, "well")
    )
    return prob_cols


def aggregate_beat_roi_well(
    df: pd.DataFrame,
    n_classes: int,
    prob_cols: Sequence[str] | None = None,
    true_col: str = "y_true",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prob_cols = validate_prediction_table(df, n_classes, prob_cols, true_col)
    agg_optional = {}
    for col in ["cond_raw", "class", "class_id"]:
        if col in df.columns:
            agg_optional[col] = (col, _first_optional)

    df_roi = (
        df.groupby(["task_name", "well_id", "roi_id"], as_index=False)
        .agg(y_true=(true_col, lambda s: _first_consistent(s, "ROI")), **agg_optional, **{c: (c, "mean") for c in prob_cols})
    )
    df_roi["y_pred"] = df_roi[prob_cols].to_numpy(dtype=float).argmax(axis=1).astype(int)

    well_optional = {}
    for col in ["cond_raw", "class", "class_id"]:
        if col in df_roi.columns:
            well_optional[col] = (col, _first_optional)
    df_well = (
        df_roi.groupby(["task_name", "well_id"], as_index=False)
        .agg(y_true=("y_true", lambda s: _first_consistent(s, "well")), **well_optional, **{c: (c, "mean") for c in prob_cols})
    )
    df_well["y_pred"] = df_well[prob_cols].to_numpy(dtype=float).argmax(axis=1).astype(int)
    return df_roi, df_well


def classification_summary(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] = DEFAULT_LABELS,
) -> dict:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", labels=list(labels), zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=list(labels)).astype(int).tolist(),
        "labels": [int(v) for v in labels],
        "n": int(len(y_true_arr)),
    }


def write_hierarchical_outputs(
    df: pd.DataFrame,
    out_dir: str | Path,
    n_classes: int,
    labels: Sequence[int] = DEFAULT_LABELS,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    roi, well = aggregate_beat_roi_well(df, n_classes=n_classes)
    roi.to_csv(out / "roi_predictions.csv", index=False)
    well.to_csv(out / "well_predictions.csv", index=False)
    summary = classification_summary(well["y_true"], well["y_pred"], labels=labels)
    with open(out / "well_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def bootstrap_macro_f1(
    df_well: pd.DataFrame,
    n_resamples: int = 5000,
    seed: int = 0,
    y_col: str = "y_true",
    pred_col: str = "y_pred",
    labels: Sequence[int] = DEFAULT_LABELS,
    stratified: bool = False,
) -> dict:
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    y = df_well[y_col].to_numpy(dtype=int)
    pred = df_well[pred_col].to_numpy(dtype=int)
    n = len(y)
    if n == 0:
        raise ValueError("cannot bootstrap an empty well-level table")
    point = float(f1_score(y, pred, average="macro", labels=list(labels), zero_division=0))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples, dtype=float)
    if stratified:
        by_class = {lab: np.flatnonzero(y == lab) for lab in np.unique(y)}
        for b in range(n_resamples):
            sample_parts = [
                rng.choice(idx, size=len(idx), replace=True)
                for idx in by_class.values()
                if len(idx) > 0
            ]
            sample_idx = np.concatenate(sample_parts)
            boots[b] = f1_score(y[sample_idx], pred[sample_idx], average="macro", labels=list(labels), zero_division=0)
    else:
        idx = np.arange(n)
        for b in range(n_resamples):
            sample_idx = rng.choice(idx, size=n, replace=True)
            boots[b] = f1_score(y[sample_idx], pred[sample_idx], average="macro", labels=list(labels), zero_division=0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "macro_f1": point,
        "ci95": [float(lo), float(hi)],
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "stratified": bool(stratified),
        "n_well": int(n),
    }
