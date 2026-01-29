"""
Evaluation metrics for M5 forecasting.

Primary metric: WRMSSE (Weighted Root Mean Squared Scaled Error)
Secondary metrics: MAE, RMSE

This module also supports evaluating saved prediction parquet files by
joining them with the processed dataset to obtain y_actual.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# -----------------------------
# Core metric code (unchanged)
# -----------------------------

def compute_scale(df: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """
    Compute scale factor for each item-store series.

    Formula:
        scale_i = sqrt( mean( (y_t - y_{t-1})^2 ) )
    """
    train = df[df["day_index"] <= cutoff_day].copy()
    train = train.sort_values(["item_id", "store_id", "day_index"])

    train["sales_lag1"] = train.groupby(["item_id", "store_id"])["sales"].shift(1)
    train["sq_diff"] = (train["sales"] - train["sales_lag1"]) ** 2

    scale_df = (
        train
        .groupby(["item_id", "store_id"])["sq_diff"]
        .mean()
        .reset_index(name="mean_sq_diff")
    )
    scale_df["scale"] = np.sqrt(scale_df["mean_sq_diff"])
    scale_df["scale"] = scale_df["scale"].replace(0, 1e-6)

    return scale_df[["item_id", "store_id", "scale"]]


def compute_weights(df: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """
    Compute revenue-based weights for each item-store.
    """
    recent = df[
        (df["day_index"] > cutoff_day - 28) &
        (df["day_index"] <= cutoff_day)
    ].copy()

    if "sell_price" in recent.columns:
        recent["revenue"] = recent["sales"] * recent["sell_price"].fillna(1)
    else:
        recent["revenue"] = recent["sales"]

    weight_df = (
        recent
        .groupby(["item_id", "store_id"])["revenue"]
        .sum()
        .reset_index(name="total_revenue")
    )

    total = weight_df["total_revenue"].sum()
    if total > 0:
        weight_df["weight"] = weight_df["total_revenue"] / total
    else:
        weight_df["weight"] = 1 / len(weight_df)

    return weight_df[["item_id", "store_id", "weight"]]


def compute_wrmsse(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    cutoff_day: int
) -> tuple[float, pd.DataFrame]:
    """
    Compute WRMSSE (Weighted Root Mean Squared Scaled Error).

    Args:
        predictions: DataFrame with [item_id, store_id, day_index, y_pred, y_actual]
        df: Full processed dataset
        cutoff_day: Last day of training data
    """
    scale_df = compute_scale(df, cutoff_day)
    weight_df = compute_weights(df, cutoff_day)

    preds_scaled = predictions.merge(scale_df, on=["item_id", "store_id"], how="left")

    preds_scaled["scaled_sq_error"] = (
        (preds_scaled["y_actual"] - preds_scaled["y_pred"]) / preds_scaled["scale"]
    ) ** 2

    rmsse_df = (
        preds_scaled
        .groupby(["item_id", "store_id"])["scaled_sq_error"]
        .mean()
        .reset_index(name="mean_sse")
    )
    rmsse_df["rmsse"] = np.sqrt(rmsse_df["mean_sse"])

    rmsse_df = rmsse_df.merge(weight_df, on=["item_id", "store_id"], how="left")
    rmsse_df["weight"] = rmsse_df["weight"].fillna(0)

    wrmsse_value = (rmsse_df["rmsse"] * rmsse_df["weight"]).sum()
    return wrmsse_value, rmsse_df[["item_id", "store_id", "rmsse", "weight"]]


def compute_mae(predictions: pd.DataFrame) -> float:
    """Mean Absolute Error."""
    valid = predictions.dropna(subset=["y_pred", "y_actual"])
    if len(valid) == 0:
        return np.nan
    return float((valid["y_actual"] - valid["y_pred"]).abs().mean())


def compute_rmse(predictions: pd.DataFrame) -> float:
    """Root Mean Squared Error."""
    valid = predictions.dropna(subset=["y_pred", "y_actual"])
    if len(valid) == 0:
        return np.nan
    return float(np.sqrt(((valid["y_actual"] - valid["y_pred"]) ** 2).mean()))


def evaluate_predictions(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    cutoff_day: int
) -> dict:
    """
    Compute all evaluation metrics for a set of predictions.
    """
    valid = predictions.dropna(subset=["y_pred", "y_actual"])

    mae = compute_mae(predictions)
    rmse = compute_rmse(predictions)
    wrmsse, _ = compute_wrmsse(valid, df, cutoff_day)

    return {
        "cutoff_day": cutoff_day,
        "mae": mae,
        "rmse": rmse,
        "wrmsse": wrmsse,
        "n_predictions": len(predictions),
        "n_valid": len(valid),
    }


# ---------------------------------------
# Additions: evaluate from parquet artifacts
# ---------------------------------------

def load_processed_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load the processed M5 dataset (parquet)."""
    return pd.read_parquet(Path(data_path))


def load_predictions_artifact(predictions_path: str | Path) -> pd.DataFrame:
    """Load predictions artifact (parquet)."""
    return pd.read_parquet(Path(predictions_path))


def ensure_actuals(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    *,
    target_col: str = "sales",
) -> pd.DataFrame:
    """
    Ensure predictions has y_actual. If missing, merge from df using sales.
    Expects df has [item_id, store_id, day_index, sales].
    """
    if "y_actual" in predictions.columns:
        return predictions

    actuals = df[["item_id", "store_id", "day_index", target_col]].rename(columns={target_col: "y_actual"})
    return predictions.merge(actuals, on=["item_id", "store_id", "day_index"], how="left")


def evaluate_from_artifacts(
    *,
    data_path: str | Path,
    predictions_path: str | Path,
    cutoff_day: int,
    horizon: Optional[int] = 28,
) -> dict:
    """
    Load df + predictions from parquet artifacts, ensure y_actual exists, optionally
    restrict to the forecast window, then compute metrics.
    """
    df = load_processed_dataset(data_path)
    preds = load_predictions_artifact(predictions_path)
    preds = ensure_actuals(preds, df)

    # Optional window filter (recommended safety)
    if horizon is not None:
        preds = preds[(preds["day_index"] > cutoff_day) & (preds["day_index"] <= cutoff_day + horizon)].copy()

    return evaluate_predictions(preds, df, cutoff_day=cutoff_day)


# ---------------------------------------
# Tyro CLI for hygiene (optional, but handy)
# ---------------------------------------

@dataclass
class Args:
    # Path to processed dataset parquet
    data_path: Path

    # Path to predictions parquet (either with y_actual already, or y_pred-only)
    predictions_path: Path

    cutoff_day: int
    horizon: Optional[int] = 28  # set to None to skip window filtering


def _main(args: Args) -> None:
    metrics = evaluate_from_artifacts(
        data_path=args.data_path,
        predictions_path=args.predictions_path,
        cutoff_day=args.cutoff_day,
        horizon=args.horizon,
    )
    print(metrics)


if __name__ == "__main__":
    import tyro
    _main(tyro.cli(Args))
