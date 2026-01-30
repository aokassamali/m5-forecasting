"""
LightGBM forecasting model for M5.

Based on 1st place solution approach:
- Simple lag/rolling features
- No categorical features (avoids train/forecast mismatch issues)
- Tweedie objective for zero-inflated count data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb


@dataclass
class LGBMConfig:
    """LightGBM hyperparameters."""
    objective: str = "tweedie"
    tweedie_variance_power: float = 1.1
    
    num_leaves: int = 255
    learning_rate: float = 0.03
    n_estimators: int = 200
    min_child_samples: int = 20
    
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1


def lgbm_forecast(
    df: pd.DataFrame,
    cutoff_day: int,
    horizon: int = 28,
    *,
    model_cfg: Optional[LGBMConfig] = None,
    **kwargs,  # Accept extra kwargs for compatibility
) -> pd.DataFrame:
    """
    Train LightGBM and forecast.
    
    Args:
        df: Processed dataset with [item_id, store_id, day_index, sales, ...]
        cutoff_day: Last day of training data
        horizon: Days to forecast
        model_cfg: Model hyperparameters
    
    Returns:
        DataFrame with [item_id, store_id, day_index, y_pred]
    """
    if model_cfg is None:
        model_cfg = LGBMConfig()
    
    print(f"[lgbm] Building features for cutoff={cutoff_day}, horizon={horizon}...")
    
    # Sort data
    df = df.sort_values(["item_id", "store_id", "day_index"]).reset_index(drop=True)
    
    # Build simple features
    g = df.groupby(["item_id", "store_id"], sort=False)
    
    # Lag features (all >= horizon to avoid leakage)
    df["lag_28"] = g["sales"].shift(28)
    df["lag_35"] = g["sales"].shift(35)
    df["lag_42"] = g["sales"].shift(42)
    df["lag_49"] = g["sales"].shift(49)
    df["lag_56"] = g["sales"].shift(56)
    
    # Rolling features (shifted by horizon)
    df["roll_mean_7"] = g["sales"].transform(lambda x: x.shift(28).rolling(7, min_periods=1).mean())
    df["roll_mean_14"] = g["sales"].transform(lambda x: x.shift(28).rolling(14, min_periods=1).mean())
    df["roll_mean_28"] = g["sales"].transform(lambda x: x.shift(28).rolling(28, min_periods=1).mean())
    df["roll_std_7"] = g["sales"].transform(lambda x: x.shift(28).rolling(7, min_periods=1).std()).fillna(0)
    df["roll_std_28"] = g["sales"].transform(lambda x: x.shift(28).rolling(28, min_periods=1).std()).fillna(0)
    
    # Calendar features (numeric only)
    df["wday"] = df["wday"].astype(int)
    df["month"] = df["month"].astype(int)
    df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"]).astype(int)
    
    # SNAP (numeric)
    state = df["state_id"].astype(str)
    df["snap"] = np.where(
        state == "CA", df["snap_CA"],
        np.where(state == "TX", df["snap_TX"], df["snap_WI"])
    ).astype(int)
    
    # Price features
    df["price"] = df["sell_price"]
    df["price_lag_7"] = g["sell_price"].shift(7)
    df["price_change"] = (df["sell_price"] / df["price_lag_7"].replace(0, np.nan) - 1).fillna(0)
    
    # Feature columns (all numeric, no categoricals)
    feature_cols = [
        "lag_28", "lag_35", "lag_42", "lag_49", "lag_56",
        "roll_mean_7", "roll_mean_14", "roll_mean_28",
        "roll_std_7", "roll_std_28",
        "wday", "month", "is_weekend", "snap",
        "price", "price_change",
    ]
    
    # Split train/forecast
    train = df[df["day_index"] <= cutoff_day].copy()
    forecast = df[(df["day_index"] > cutoff_day) & (df["day_index"] <= cutoff_day + horizon)].copy()
    
    print(f"[lgbm] Train rows before dropna: {len(train):,}")
    
    # Drop NaN in train
    train_clean = train.dropna(subset=feature_cols + ["sales"])
    print(f"[lgbm] Train rows after dropna: {len(train_clean):,}")
    
    if len(train_clean) == 0:
        raise ValueError("No training data after dropna!")
    
    X_train = train_clean[feature_cols]
    y_train = train_clean["sales"]
    
    # For forecast, fill NaN with 0 (these are edge cases)
    X_forecast = forecast[feature_cols].fillna(0)
    
    print(f"[lgbm] Forecast rows: {len(X_forecast):,}")
    
    # Train model
    print(f"[lgbm] Training with {model_cfg.objective} objective...")
    model = lgb.LGBMRegressor(
        objective=model_cfg.objective,
        tweedie_variance_power=model_cfg.tweedie_variance_power,
        num_leaves=model_cfg.num_leaves,
        learning_rate=model_cfg.learning_rate,
        n_estimators=model_cfg.n_estimators,
        min_child_samples=model_cfg.min_child_samples,
        subsample=model_cfg.subsample,
        colsample_bytree=model_cfg.colsample_bytree,
        random_state=model_cfg.random_state,
        n_jobs=model_cfg.n_jobs,
        verbose=model_cfg.verbose,
    )
    model.fit(X_train, y_train)
    
    # Predict
    print("[lgbm] Predicting...")
    y_pred = model.predict(X_forecast)
    y_pred = np.maximum(y_pred, 0.0)  # Clip negatives
    
    print(f"[lgbm] Predictions: min={y_pred.min():.2f}, mean={y_pred.mean():.2f}, max={y_pred.max():.2f}")
    
    # Build output
    out = forecast[["item_id", "store_id", "day_index"]].copy()
    out["y_pred"] = y_pred
    
    return out