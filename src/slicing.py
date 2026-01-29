"""
Error slicing for M5 forecasting.

Categorizes item-stores into tiers for disaggregated evaluation:
- Volume tiers (Low/Medium/High based on total sales in TRAINING window)
- Intermittency tiers (Low/Medium/High based on % zero-sales days in TRAINING window)

All tier assignments use only training data (day_index <= cutoff_day) to avoid leakage.

This module is intended to be "evaluation glue":
- It does not write files.
- It returns DataFrames / dicts that orchestration code (e.g., run_backtest.py) can serialize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from evaluation import compute_scale, compute_weights, evaluate_predictions


TIER_LABELS = ("Low", "Medium", "High")


def _training_slice(df: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """Return training-window rows (no leakage)."""
    return df[df["day_index"] <= cutoff_day].copy()


def compute_volume_tiers(
    df: pd.DataFrame,
    cutoff_day: int,
    n_tiers: int = 3,
) -> pd.DataFrame:
    """
    Assign each item-store to a volume tier based on total training sales.

    Returns:
        DataFrame with columns [item_id, store_id, total_sales, volume_tier]
    """
    if n_tiers < 1:
        raise ValueError("n_tiers must be >= 1")

    train = _training_slice(df, cutoff_day)

    volume_df = (
        train.groupby(["item_id", "store_id"])["sales"]
        .sum()
        .reset_index(name="total_sales")
    )

    if n_tiers == 1:
        volume_df["volume_tier"] = "All"
        return volume_df[["item_id", "store_id", "total_sales", "volume_tier"]]

    labels = list(TIER_LABELS[:n_tiers])

    # qcut can drop bins if ties dominate; duplicates="drop" avoids hard errors.
    volume_df["volume_tier"] = pd.qcut(
        volume_df["total_sales"],
        q=n_tiers,
        labels=labels,
        duplicates="drop",
    )

    # If qcut collapses everything to NaN (rare but possible), fallback to single tier.
    if volume_df["volume_tier"].isna().all():
        volume_df["volume_tier"] = "All"

    return volume_df[["item_id", "store_id", "total_sales", "volume_tier"]]


def compute_intermittency_tiers(
    df: pd.DataFrame,
    cutoff_day: int,
    n_tiers: int = 3,
) -> pd.DataFrame:
    """
    Assign each item-store to an intermittency tier based on % zero-sales days in training.

    Returns:
        DataFrame with columns [item_id, store_id, zero_pct, intermittency_tier]
    """
    if n_tiers < 1:
        raise ValueError("n_tiers must be >= 1")

    train = _training_slice(df, cutoff_day)

    g = train.groupby(["item_id", "store_id"])["sales"]
    intermit_df = g.agg(
        n_days="count",
        n_zeros=lambda x: (x == 0).sum(),
    ).reset_index()

    intermit_df["zero_pct"] = intermit_df["n_zeros"] / intermit_df["n_days"].replace(0, np.nan)
    intermit_df["zero_pct"] = intermit_df["zero_pct"].fillna(0.0)

    if n_tiers == 1:
        intermit_df["intermittency_tier"] = "All"
        return intermit_df[["item_id", "store_id", "zero_pct", "intermittency_tier"]]

    labels = list(TIER_LABELS[:n_tiers])
    intermit_df["intermittency_tier"] = pd.qcut(
        intermit_df["zero_pct"],
        q=n_tiers,
        labels=labels,
        duplicates="drop",
    )

    if intermit_df["intermittency_tier"].isna().all():
        intermit_df["intermittency_tier"] = "All"

    return intermit_df[["item_id", "store_id", "zero_pct", "intermittency_tier"]]


def add_tiers_to_predictions(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    cutoff_day: int,
    *,
    volume_tiers: bool = True,
    intermittency_tiers: bool = True,
    n_tiers: int = 3,
) -> pd.DataFrame:
    """
    Add tier columns to predictions.

    Args:
        predictions: DataFrame with at least [item_id, store_id]
        df: Full dataset (used to compute tiers from training window only)
        cutoff_day: last day of training window
        volume_tiers: whether to add volume_tier
        intermittency_tiers: whether to add intermittency_tier
        n_tiers: number of tiers for both schemes

    Returns:
        predictions with added columns as requested
    """
    preds = predictions.copy()

    if volume_tiers:
        vol = compute_volume_tiers(df, cutoff_day, n_tiers=n_tiers)
        preds = preds.merge(vol[["item_id", "store_id", "volume_tier"]], on=["item_id", "store_id"], how="left")

    if intermittency_tiers:
        it = compute_intermittency_tiers(df, cutoff_day, n_tiers=n_tiers)
        preds = preds.merge(
            it[["item_id", "store_id", "intermittency_tier"]],
            on=["item_id", "store_id"],
            how="left",
        )

    return preds


def compute_wrmsse_by_slice(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    cutoff_day: int,
    slice_col: str,
) -> pd.DataFrame:
    """
    Compute slice-level metrics.

    Notes:
      - Uses item_id x store_id series.
      - Computes per-series RMSSE using the global scale (training window only).
      - Uses revenue weights computed on the global training window, then RENORMALIZES
        weights within each slice (so each slice's WRMSSE is comparable within itself).

    Args:
        predictions: DataFrame with [item_id, store_id, y_pred, y_actual, slice_col]
        df: Full dataset (for scale/weight computation)
        cutoff_day: training cutoff day
        slice_col: column to slice by (e.g., volume_tier or intermittency_tier)

    Returns:
        DataFrame with one row per slice value:
        [slice_col, tier, cutoff_day, wrmsse, mae, rmse, n_series, n_predictions, n_valid]
    """
    required = {"item_id", "store_id", "y_pred", "y_actual", slice_col}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing required columns: {sorted(missing)}")

    # Compute global scale + weights (training-window only)
    scale_df = compute_scale(df, cutoff_day)
    weight_df = compute_weights(df, cutoff_day)

    preds = predictions.merge(scale_df, on=["item_id", "store_id"], how="left")
    preds = preds.merge(weight_df, on=["item_id", "store_id"], how="left")

    # Only evaluate valid rows
    preds_valid = preds.dropna(subset=["y_pred", "y_actual"]).copy()

    # Avoid division by zero just in case (compute_scale already guards)
    preds_valid["scale"] = preds_valid["scale"].fillna(1e-6).replace(0, 1e-6)

    preds_valid["err"] = preds_valid["y_actual"] - preds_valid["y_pred"]
    preds_valid["sq_err"] = preds_valid["err"] ** 2
    preds_valid["abs_err"] = preds_valid["err"].abs()
    preds_valid["scaled_sq_error"] = (preds_valid["err"] / preds_valid["scale"]) ** 2

    results: list[dict] = []

    for tier in sorted(preds_valid[slice_col].dropna().unique()):
        slice_preds = preds_valid[preds_valid[slice_col] == tier]

        # Per-series RMSSE within slice
        per_series = (
            slice_preds
            .groupby(["item_id", "store_id"])
            .agg(
                mean_scaled_sse=("scaled_sq_error", "mean"),
                weight=("weight", "first"),
            )
            .reset_index()
        )
        per_series["rmsse"] = np.sqrt(per_series["mean_scaled_sse"])

        # Renormalize weights within slice
        total_w = per_series["weight"].sum()
        if total_w > 0:
            per_series["weight_norm"] = per_series["weight"] / total_w
        else:
            per_series["weight_norm"] = 1 / len(per_series)

        wrmsse = float((per_series["rmsse"] * per_series["weight_norm"]).sum())
        mae = float(slice_preds["abs_err"].mean())
        rmse = float(np.sqrt(slice_preds["sq_err"].mean()))

        results.append(
            {
                "slice_col": slice_col,
                "tier": str(tier),
                "cutoff_day": int(cutoff_day),
                "wrmsse": wrmsse,
                "mae": mae,
                "rmse": rmse,
                "n_series": int(len(per_series)),
                "n_predictions": int(len(slice_preds)),
                "n_valid": int(len(slice_preds)),
            }
        )

    return pd.DataFrame(results)


def evaluate_with_slices(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    cutoff_day: int,
    *,
    n_tiers: int = 3,
) -> dict:
    """
    Evaluate overall + sliced metrics.

    Returns:
        dict with:
          - "overall": overall metrics dict (mae/rmse/wrmsse/...)
          - "by_volume": DataFrame of slice metrics by volume_tier
          - "by_intermittency": DataFrame of slice metrics by intermittency_tier
    """
    preds_with_tiers = add_tiers_to_predictions(
        predictions,
        df,
        cutoff_day,
        volume_tiers=True,
        intermittency_tiers=True,
        n_tiers=n_tiers,
    )

    overall = evaluate_predictions(predictions, df, cutoff_day)

    by_volume = compute_wrmsse_by_slice(preds_with_tiers, df, cutoff_day, "volume_tier")
    by_intermittency = compute_wrmsse_by_slice(preds_with_tiers, df, cutoff_day, "intermittency_tier")

    return {
        "overall": overall,
        "by_volume": by_volume,
        "by_intermittency": by_intermittency,
    }
