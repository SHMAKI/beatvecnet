#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from beatvecnet.dataset import DatasetCfg, SampleDataset, load_manifest
from beatvecnet.evaluation import has_hierarchy_metadata, write_hierarchical_outputs
from beatvecnet.model import BeatVecNet, ModelCfg
from beatvecnet.transforms import BeatTransform, ChannelStandardization


def build_dataset_cfg(cfg: dict) -> DatasetCfg:
    return DatasetCfg(
        n_classes=int(cfg["data"]["n_classes"]),
        size_y=int(cfg["data"]["size_y"]),
        size_t=int(cfg["data"]["size_t"]),
        in_ch=int(cfg["data"]["in_ch"]),
        mmap_mode=cfg["data"].get("mmap_mode", "r"),
        dtype=torch.float32,
        metadata_columns=cfg["data"].get("metadata_columns"),
    )


def build_model(cfg: dict, device: torch.device):
    model_cfg = ModelCfg(
        n_output=int(cfg["model"]["n_output"]),
        n_layers=int(cfg["model"]["n_layers"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        mid_units=int(cfg["model"]["mid_units"]),
        dropout_rate=float(cfg["model"]["dropout_rate"]),
        n_filters=tuple(cfg["model"]["n_filters"]),
        n_channel_in=int(cfg["model"]["n_channel_in"]),
    )
    return BeatVecNet(model_cfg).to(device)


def extract_standardization(ckpt: dict) -> tuple[list[float], list[float]]:
    mean = ckpt.get("sample_training_channel_mean")
    std = ckpt.get("sample_training_channel_std")
    if mean is None or std is None:
        raise KeyError(
            "checkpoint does not contain sample_training_channel_mean/std; "
            "refusing to estimate normalization from validation/test data"
        )
    return [float(v) for v in mean], [float(v) for v in std]


def build_loader(cfg: dict, split: str, mean: list[float], std: list[float]):
    ds_cfg = build_dataset_cfg(cfg)
    df = load_manifest(cfg["data"]["manifest_path"], split=split)
    if len(df) == 0:
        raise ValueError(f"manifest split is empty: {split}")
    ds = SampleDataset(
        df=df,
        cfg=ds_cfg,
        repo_root=cfg["data"].get("repo_root", "."),
        path_col=cfg["data"].get("path_col", "path"),
        transform=BeatTransform(ChannelStandardization(mean=mean, std=std), None, train=False),
        return_meta=True,
        enforce_shape=True,
    )
    return df, DataLoader(
        ds,
        batch_size=int(cfg["test"].get("batch_size", cfg["train"].get("validation_batch_size", cfg["train"].get("batch_size", 64)))),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def predict(model, loader, device: torch.device, criterion: nn.Module | None = None):
    model.eval()
    probs_all, pred_all, true_all, meta_rows = [], [], [], []
    total_loss = 0.0
    total_n = 0
    for x, y, meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * int(y.size(0))
            total_n += int(y.size(0))
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        probs_all.append(probs.cpu().numpy())
        pred_all.append(pred.cpu().numpy())
        true_all.append(y.cpu().numpy())
        for i in range(len(pred)):
            row = {}
            for k, v in meta.items():
                if isinstance(v, torch.Tensor):
                    row[k] = v[i].item()
                else:
                    row[k] = v[i]
            meta_rows.append(row)
    return (
        np.concatenate(probs_all, axis=0),
        np.concatenate(pred_all, axis=0),
        np.concatenate(true_all, axis=0),
        meta_rows,
        total_loss / max(total_n, 1) if criterion is not None else None,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--hierarchical", action="store_true", help="Run hierarchical evaluation when metadata exists.")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg["test"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint or cfg["test"]["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    mean, std = extract_standardization(ckpt)
    df_split, loader = build_loader(cfg, args.split, mean, std)
    model = build_model(ckpt.get("config", cfg), device)
    model.load_state_dict(ckpt["model_state_dict"])

    n_classes = int(cfg["data"]["n_classes"])
    class_names = ckpt.get("class_names", cfg.get("test", {}).get("class_names", ["control", "doxorubicin", "trastuzumab", "sunitinib"]))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["train"].get("label_smoothing", 0.0)))
    probs, pred, true, meta_rows, loss = predict(model, loader, device, criterion=criterion)

    df_pred = pd.DataFrame(meta_rows) if meta_rows else df_split.copy()
    df_pred["y_true"] = true
    df_pred["y_pred"] = pred
    for i in range(n_classes):
        df_pred[f"prob_{i}"] = probs[:, i]
    df_pred.to_csv(out_dir / f"{args.split}_sample_predictions.csv", index=False)

    labels = list(range(n_classes))
    cm = confusion_matrix(true, pred, labels=labels)
    np.save(out_dir / f"{args.split}_sample_confusion_matrix.npy", cm)
    metrics = {
        "split": args.split,
        "loss": float(loss) if loss is not None else None,
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", labels=labels, zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
    }
    with open(out_dir / f"{args.split}_sample_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    report = classification_report(
        true,
        pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(out_dir / f"{args.split}_sample_classification_report.csv")

    if args.hierarchical:
        if has_hierarchy_metadata(df_pred):
            summary = write_hierarchical_outputs(df_pred, out_dir / "hierarchical", n_classes=n_classes, labels=labels)
            print("Hierarchical well-level results")
            print(json.dumps(summary, indent=2))
        else:
            print("Hierarchy metadata not found; wrote sample-level outputs only.")
    print("Sample-level results")
    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
