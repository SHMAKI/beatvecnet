#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from beatvecnet.evaluation import bootstrap_macro_f1, write_hierarchical_outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate beat predictions to ROI and well levels.")
    ap.add_argument("--predictions", required=True, help="Beat-level prediction CSV")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--bootstrap-resamples", type=int, default=0)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--bootstrap-stratified", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    out_dir = Path(args.out_dir)
    summary = write_hierarchical_outputs(df, out_dir=out_dir, n_classes=args.n_classes, labels=list(range(args.n_classes)))
    if args.bootstrap_resamples > 0:
        well = pd.read_csv(out_dir / "well_predictions.csv")
        boot = bootstrap_macro_f1(
            well,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
            labels=list(range(args.n_classes)),
            stratified=args.bootstrap_stratified,
        )
        with open(out_dir / "well_macro_f1_bootstrap.json", "w") as f:
            json.dump(boot, f, indent=2)
        summary["bootstrap"] = boot
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
