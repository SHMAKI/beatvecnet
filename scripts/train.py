#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from beatvecnet.dataset import DatasetCfg, SampleDataset, load_manifest
from beatvecnet.model import BeatVecNet, ModelCfg
from beatvecnet.standardization import compute_channel_mean_std, save_standardization_json
from beatvecnet.transforms import BeatTransform, ChannelStandardization, RandomTemporalShift


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_names(cfg: dict) -> list[str]:
    return cfg.get("data", {}).get(
        "class_names",
        cfg.get("test", {}).get("class_names", ["control", "doxorubicin", "trastuzumab", "sunitinib"]),
    )


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


def build_model(cfg: dict, device: torch.device) -> nn.Module:
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


def make_amp(device: torch.device, amp_mode: str | bool):
    enabled = bool(device.type == "cuda" and str(amp_mode).lower() in {"auto", "true", "1", "yes"})
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=enabled)
        return enabled, scaler
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        scaler = torch.cuda.amp.GradScaler(enabled=enabled)
    else:
        raise RuntimeError("this PyTorch build does not provide GradScaler")
    return enabled, scaler


def autocast_context(device: torch.device, enabled: bool):
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=enabled)


def build_dataloaders(cfg: dict, mean: np.ndarray, std: np.ndarray):
    manifest_path = cfg["data"]["manifest_path"]
    repo_root = cfg["data"].get("repo_root", ".")
    num_workers = int(cfg["train"].get("num_workers", 0))
    train_batch = int(cfg["train"]["batch_size"])
    val_batch = int(cfg["train"].get("validation_batch_size", train_batch))
    ds_cfg = build_dataset_cfg(cfg)

    df_train = load_manifest(manifest_path, split="train")
    df_val = load_manifest(manifest_path, split="val")
    if len(df_train) == 0 or len(df_val) == 0:
        raise ValueError("manifest must contain non-empty train and val splits")

    standardization = ChannelStandardization(mean=mean.tolist(), std=std.tolist())
    tshift_cfg = cfg.get("augmentation", {}).get("temporal_shift", {})
    train_shift = None
    if bool(tshift_cfg.get("enabled", False)):
        train_shift = RandomTemporalShift(
            maximum_shift_fraction=float(tshift_cfg.get("maximum_shift_fraction", 0.05)),
            application_probability=float(tshift_cfg.get("application_probability", 0.5)),
            boundary_fill=str(tshift_cfg.get("boundary_fill", "zero")),
        )

    train_ds = SampleDataset(
        df=df_train,
        cfg=ds_cfg,
        repo_root=repo_root,
        path_col=cfg["data"].get("path_col", "path"),
        transform=BeatTransform(standardization, train_shift, train=True),
        return_meta=True,
        enforce_shape=True,
    )
    val_ds = SampleDataset(
        df=df_val,
        cfg=ds_cfg,
        repo_root=repo_root,
        path_col=cfg["data"].get("path_col", "path"),
        transform=BeatTransform(standardization, None, train=False),
        return_meta=True,
        enforce_shape=True,
    )
    common = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available(), "drop_last": False}
    return (
        DataLoader(train_ds, batch_size=train_batch, shuffle=True, **common),
        DataLoader(val_ds, batch_size=val_batch, shuffle=False, **common),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    amp_enabled: bool = False,
    grad_clip_max_norm: float | None = None,
    labels: list[int] | None = None,
) -> dict:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_n = 0
    preds: list[int] = []
    truths: list[int] = []

    for batch in loader:
        x, y = batch[:2]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with autocast_context(device, amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                assert scaler is not None
                scaler.scale(loss).backward()
                if grad_clip_max_norm is not None and grad_clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_max_norm))
                    if not torch.isfinite(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        continue
                scaler.step(optimizer)
                scaler.update()
        pred = logits.argmax(dim=1)
        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        total_n += bs
        preds.extend(pred.detach().cpu().numpy().astype(int).tolist())
        truths.extend(y.detach().cpu().numpy().astype(int).tolist())

    labels = labels or list(range(int(max(truths) + 1))) if truths else [0, 1, 2, 3]
    return {
        "loss": total_loss / max(total_n, 1),
        "accuracy": float(accuracy_score(truths, preds)) if truths else 0.0,
        "macro_f1": float(f1_score(truths, preds, average="macro", labels=labels, zero_division=0)) if truths else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--epochs", type=int, default=None, help="Optional epoch override for smoke tests")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["train"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ds_cfg = build_dataset_cfg(cfg)
    df_train = load_manifest(cfg["data"]["manifest_path"], split="train")
    mean, std = compute_channel_mean_std(
        df_train,
        cfg=ds_cfg,
        repo_root=cfg["data"].get("repo_root", "."),
        path_col=cfg["data"].get("path_col", "path"),
    )
    save_standardization_json(out_dir / "sample_training_standardization.json", mean, std)

    train_loader, val_loader = build_dataloaders(cfg, mean=mean, std=std)
    model = build_model(cfg, device)
    labels = list(range(int(cfg["data"]["n_classes"])))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["train"].get("label_smoothing", 0.0)))
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("learning_rate", cfg["train"].get("lr", 1e-3))),
        weight_decay=float(opt_cfg.get("weight_decay", cfg["train"].get("weight_decay", 1e-4))),
        betas=(float(opt_cfg.get("beta1", 0.9)), float(opt_cfg.get("beta2", 0.999))),
        eps=float(opt_cfg.get("epsilon", 1e-8)),
    )
    amp_enabled, scaler = make_amp(device, cfg["train"].get("amp", "auto"))
    early = cfg["train"].get("early_stopping", {})
    patience = int(early.get("patience", 20)) if bool(early.get("enabled", True)) else int(cfg["train"]["epochs"])
    delta = float(early.get("delta", 0.0))

    best_f1 = -1.0
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip_max_norm=float(cfg["train"].get("grad_clip_max_norm", 0.0)),
            labels=labels,
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, labels=labels)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"[{epoch:03d}] train_loss={train_metrics['loss']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        improved = val_metrics["macro_f1"] > best_f1 + delta
        if improved:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_macro_f1": float(best_f1),
                    "config": cfg,
                    "sample_training_channel_mean": mean.tolist(),
                    "sample_training_channel_std": std.tolist(),
                    "class_names": class_names(cfg),
                },
                out_dir / "best.pt",
            )
        else:
            stale += 1
        if stale >= patience:
            print(f"Early stopping after {patience} epochs without validation macro-F1 improvement.")
            break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump({"best_epoch": best_epoch, "best_validation_macro_f1": best_f1, "device": str(device)}, f, indent=2)
    print(f"Best validation macro-F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
