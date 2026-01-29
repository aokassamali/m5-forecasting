"""
Run evaluation with error slicing.

Usage:
  # Load predictions from artifact
  python scripts/run_sliced_eval.py `
    --data-path data/processed/m5_melted_merged.parquet `
    --preds-path results/predictions_df/lgbm_cutoff_1913_20260128_194457.parquet `
    --cutoff-day 1913

  # Or generate seasonal naive predictions on the fly
  python scripts/run_sliced_eval.py `
    --data-path data/processed/m5_melted_merged.parquet `
    --model naive `
    --cutoff-day 1913
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import tyro

from models.naive import seasonal_naive_predict
from slicing import evaluate_with_slices


@dataclass
class Args:
    data_path: Path = Path("data/processed/m5_melted_merged.parquet")
    cutoff_day: int = 1913

    # If provided, load predictions from parquet instead of generating
    preds_path: Optional[Path] = None

    # If preds_path is None, we can generate a baseline quickly
    model: Literal["naive"] = "naive"
    horizon: int = 28
    lag: int = 7

    # Optional store filter
    stores: Optional[list[str]] = None


def ensure_actuals(preds: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if "y_actual" in preds.columns:
        return preds
    actuals = df[["item_id", "store_id", "day_index", "sales"]].copy()
    actuals = actuals.rename(columns={"sales": "y_actual"})
    return preds.merge(actuals, on=["item_id", "store_id", "day_index"], how="left")


def main(args: Args) -> None:
    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)

    if args.stores is not None:
        df = df[df["store_id"].isin(args.stores)].copy()
        print(f"Filtered to stores: {args.stores}")

    print(f"Data shape: {df.shape}")

    if args.preds_path is not None:
        print(f"Loading predictions from {args.preds_path}...")
        preds = pd.read_parquet(args.preds_path)
    else:
        print(f"Generating predictions with {args.model}...")
        if args.model == "naive":
            preds = seasonal_naive_predict(df, cutoff_day=args.cutoff_day, horizon=args.horizon, lag=args.lag)
        else:
            raise ValueError("Only naive generation supported here. Use preds_path for model artifacts.")

    preds = ensure_actuals(preds, df)

    # Keep only forecast window
    preds = preds[(preds["day_index"] > args.cutoff_day) & (preds["day_index"] <= args.cutoff_day + 28)].copy()

    print("Evaluating with slices...")
    results = evaluate_with_slices(preds, df, args.cutoff_day)

    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    for k, v in results["overall"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("BY VOLUME TIER")
    print("=" * 70)
    print(results["by_volume"].to_string(index=False))

    print("\n" + "=" * 70)
    print("BY INTERMITTENCY TIER")
    print("=" * 70)
    print(results["by_intermittency"].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main(tyro.cli(Args))
