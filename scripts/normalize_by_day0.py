#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from beatvecnet.amplitude_normalization import fit_day0_well_scales, normalize_manifest_tensors
from beatvecnet.dataset import normalize_manifest_metadata_columns


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply day 0 well-level amplitude normalization.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--path-col", default="path")
    ap.add_argument("--well-col", default="well_id")
    ap.add_argument("--day-col", default="day")
    ap.add_argument("--day0-value", default="0")
    args = ap.parse_args()

    manifest = normalize_manifest_metadata_columns(pd.read_csv(args.manifest))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = fit_day0_well_scales(
        manifest,
        repo_root=args.repo_root,
        path_col=args.path_col,
        well_col=args.well_col,
        day_col=args.day_col,
        day0_value=args.day0_value,
    )
    scales.to_csv(out_dir / "day0_well_scales.csv", index=False)
    norm_manifest = normalize_manifest_tensors(
        manifest,
        scales=scales,
        repo_root=args.repo_root,
        output_dir=out_dir,
        path_col=args.path_col,
        well_col=args.well_col,
    )
    norm_manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"Wrote normalized tensors and manifest to {out_dir}")


if __name__ == "__main__":
    main()
