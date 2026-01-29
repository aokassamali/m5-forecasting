"""
Seasonal Naive baseline model.

Prediction: y_hat[t] = y[t - lag]
Default lag=7 (weekly seasonality).

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro


def seasonal_naive_predict(
    df: pd.DataFrame,
    cutoff_day: int,
    horizon: int = 28,
    lag: int = 7,
    *,
    item_col: str = "item_id",
    store_col: str = "store_id",
    day_col: str = "day_index",
    target_col: str = "sales",
) -> pd.DataFrame:
    """
    Pure seasonal naive predictor.

    Args:
        df: DataFrame containing at least [item_id, store_id, day_index, sales]
        cutoff_day: Last day included in training.
                   Forecasts are for days (cutoff_day+1 ... cutoff_day+horizon).
        horizon: Number of forecast days.
        lag: Seasonal lag (7 = weekly).
        *_col: Column names (configurable).

    Returns:
        DataFrame with columns [item_id, store_id, day_index, y_pred]
        for all item-store pairs in df and all forecast days.
        If lookup history is missing, y_pred will be NaN for that row.
    """
    required = {item_col, store_col, day_col, target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    forecast_days = list(range(cutoff_day + 1, cutoff_day + horizon + 1))
    lookup_days = [d - lag for d in forecast_days]

    # Universe of series
    item_stores = df[[item_col, store_col]].drop_duplicates()

    # Only pull history we need
    historical = df[df[day_col].isin(lookup_days)][[item_col, store_col, day_col, target_col]].copy()

    predictions = []
    for forecast_day in forecast_days:
        lookup_day = forecast_day - lag

        lookup_sales = historical[historical[day_col] == lookup_day][
            [item_col, store_col, target_col]
        ].copy()

        lookup_sales = lookup_sales.rename(columns={target_col: "y_pred"})
        lookup_sales[day_col] = forecast_day
        predictions.append(lookup_sales)

    pred_df = pd.concat(predictions, ignore_index=True)

    # Ensure all item-store pairs appear (even if missing lookup history)
    pred_df = item_stores.merge(pred_df, on=[item_col, store_col], how="left")

    return pred_df[[item_col, store_col, day_col, "y_pred"]]


def filter_stores(df: pd.DataFrame, stores: list[str], *, store_col: str = "store_id") -> pd.DataFrame:
    """Optional helper for local testing."""
    return df[df[store_col].isin(stores)].copy()


@dataclass
class SmokeTestArgs:
    # Path to a processed parquet that already contains melted/merged data
    data_path: Path = Path("data/processed/m5_melted_merged.parquet")

    # Forecast settings
    cutoff_day: int = 1900
    horizon: int = 28
    lag: int = 7

    # Optional store filter for quick tests (e.g., --stores CA_1 CA_2)
    stores: Optional[list[str]] = None


def _smoke_test(args: SmokeTestArgs) -> None:
    df = pd.read_parquet(args.data_path)

    if args.stores is not None:
        df = filter_stores(df, args.stores)

    preds = seasonal_naive_predict(df, cutoff_day=args.cutoff_day, horizon=args.horizon, lag=args.lag)

    print(f"Predictions shape: {preds.shape}")
    print(f"Forecast days: {preds['day_index'].min()} to {preds['day_index'].max()}")
    print(preds.head(10))


if __name__ == "__main__":
    _smoke_test(tyro.cli(SmokeTestArgs))
