"""
Feature building for M5 forecasting.

We support two modeling styles:

A) Direct (multi-horizon, cutoff-anchored features)
   - Used only if you run lgbm_recursive=False.
   - Features are computed using only history <= cutoff.

B) Recursive (one-step ahead training + iterative forecasting)
   - Used when lgbm_recursive=True.
   - Train a one-step model using lag/rolling relative to each training row's day.
   - Forecast iteratively and feed predictions forward (sales), while prices/SNAP/events are treated as known.

This file provides:
- FeatureConfig + make_train_forecast_matrices  (direct)
- RecursiveFeatureConfig + make_one_step_train_matrix  (recursive training)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------------------- Helpers ---------------------------------


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")


def _add_common_exog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds leakage-safe exogenous features derived from calendar columns.
    Assumes df already contains:
      weekday, event_name_1, state_id, snap_CA/snap_TX/snap_WI
    """
    out = df.copy()

    # Weekend flag from weekday string (robust)
    out["is_weekend"] = out["weekday"].isin(["Saturday", "Sunday"]).astype(int)

    # Event presence (binary)
    out["has_event"] = out["event_name_1"].notna().astype(int)

    # State-specific SNAP (reduce noise)
    # Only one is relevant per row.
    state = out["state_id"].astype(str)
    snap = np.where(
        state == "CA", out["snap_CA"].to_numpy(),
        np.where(state == "TX", out["snap_TX"].to_numpy(), out["snap_WI"].to_numpy())
    )
    out["snap"] = pd.Series(snap, index=out.index).astype(int)

    return out


def _cast_categoricals(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


# -------------------------- Direct (optional) ---------------------------


@dataclass(frozen=True)
class FeatureConfig:
    """
    Direct multi-horizon (cutoff-anchored) features.

    This mode is not the focus now, but kept so your CLI can still run
    with --lgbm-recursive False if you want.
    """
    horizon: int = 28

    # Sales features anchored by horizon (leakage-safe for direct multi-horizon)
    lag_days: Tuple[int, ...] = (28, 56)
    roll_windows: Tuple[int, ...] = (28, 56)

    # Price features (known at forecast time)
    price_lag_days: Tuple[int, ...] = (7, 28)
    price_roll_windows: Tuple[int, ...] = (28,)

    include_event_type: bool = True
    include_intermittency_state: bool = True
    days_since_nonzero_cap: int = 999


def build_feature_frame_direct(df: pd.DataFrame, cutoff_day: int, cfg: FeatureConfig) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a feature frame for BOTH training rows (<= cutoff) and forecast window rows
    (cutoff+1..cutoff+horizon), using only history <= cutoff for sales-derived features.

    Strategy:
      - For every row, create h = day_index - cutoff_day (<=0 for training, >=1 for forecast).
      - Compute sales_shift_h = sales shifted by cfg.horizon within each series.
      - Rolling stats computed on sales_shift_h so they only use <= cutoff history for forecast rows.
    """
    required = {
        "item_id", "store_id", "day_index", "sales",
        "dept_id", "cat_id", "state_id",
        "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1",
        "snap_CA", "snap_TX", "snap_WI",
        "sell_price",
    }
    _require_columns(df, required)

    out = df.copy()
    out = out.sort_values(["item_id", "store_id", "day_index"]).reset_index(drop=True)

    # Exog
    out = _add_common_exog(out)
    out = _cast_categoricals(out, ["item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday", "event_type_1"])

    # h relative to cutoff
    out["h"] = out["day_index"] - cutoff_day

    g = out.groupby(["item_id", "store_id"], sort=False)

    # -----------------------------
    # Intermittency state features (cutoff-anchored, leakage-safe)
    # -----------------------------
    if cfg.include_intermittency_state:
        # Mask sales after cutoff so forecast rows never use future actuals
        out["_hist_sales"] = out["sales"].where(out["day_index"] <= cutoff_day, np.nan)

        # Nonzero rate over training history (constant per series)
        out["_nonzero_flag_train"] = (out["_hist_sales"] > 0).astype("float32")
        out["nonzero_rate_train"] = g["_nonzero_flag_train"].transform("mean")

        # Days since last nonzero sale (based ONLY on training history)
        out["_pos_day"] = out["day_index"].where(out["_hist_sales"] > 0)
        out["last_pos_day"] = g["_pos_day"].ffill()

        out["days_since_nonzero"] = (
            (out["day_index"] - out["last_pos_day"])
            .astype("float32")
            .fillna(float(cfg.days_since_nonzero_cap))
            .clip(lower=0.0, upper=float(cfg.days_since_nonzero_cap))
            .astype("float32")
        )

        # Cleanup helper cols
        out = out.drop(columns=["_hist_sales", "_nonzero_flag_train", "_pos_day", "last_pos_day"])




    # Sales features anchored by horizon shift (safe for forecast rows)
    sales_shift_h = g["sales"].shift(cfg.horizon)

    for lag in cfg.lag_days:
        out[f"lag_{lag}_from_h"] = g["sales"].shift(cfg.horizon + lag)

    for w in cfg.roll_windows:
        out[f"roll_mean_{w}_from_h"] = g["sales"].transform(
            lambda s: s.shift(cfg.horizon).rolling(window=w, min_periods=1).mean()
        )
        out[f"roll_std_{w}_from_h"] = g["sales"].transform(
            lambda s: s.shift(cfg.horizon).rolling(window=w, min_periods=1).std()
        )
    # Price features (known future): compute relative to row day with shift(1) to avoid using same-day price for rolling
    for pl in cfg.price_lag_days:
        out[f"price_lag_{pl}"] = g["sell_price"].shift(pl)
        out[f"price_change_{pl}"] = (out["sell_price"] / out[f"price_lag_{pl}"]) - 1.0

    for w in cfg.price_roll_windows:
        out[f"price_roll_mean_{w}"] = g["sell_price"].transform(lambda s: s.shift(1).rolling(w, min_periods=w).mean())
        out[f"price_roll_std_{w}"] = g["sell_price"].transform(lambda s: s.shift(1).rolling(w, min_periods=w).std())

    feature_cols = [
        # IDs
        "item_id", "dept_id", "cat_id", "store_id", "state_id",
        # calendar
        "wday", "month", "year", "is_weekend", "weekday",
        # SNAP / events
        "snap", "has_event",
        # horizon step (helps direct multi-horizon)
        "h",
        # price
        "sell_price",
    ]
    
    if cfg.include_intermittency_state:
        feature_cols += ["nonzero_rate_train", "days_since_nonzero"]


    feature_cols += [f"lag_{lag}_from_h" for lag in cfg.lag_days]
    for w in cfg.roll_windows:
        feature_cols += [f"roll_mean_{w}_from_h", f"roll_std_{w}_from_h"]

    for pl in cfg.price_lag_days:
        feature_cols += [f"price_lag_{pl}", f"price_change_{pl}"]
    for w in cfg.price_roll_windows:
        feature_cols += [f"price_roll_mean_{w}", f"price_roll_std_{w}"]

    return out, feature_cols


def make_train_forecast_matrices(
    df: pd.DataFrame,
    cutoff_day: int,
    cfg: FeatureConfig,
    *,
    dropna_train: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Direct matrices for training and forecasting.
    """
    feat_df, feature_cols = build_feature_frame_direct(df, cutoff_day, cfg)

    train = feat_df[feat_df["day_index"] <= cutoff_day].copy()
    forecast = feat_df[(feat_df["day_index"] > cutoff_day) & (feat_df["day_index"] <= cutoff_day + cfg.horizon)].copy()

    X_train = train[feature_cols].copy()
    y_train = train["sales"].copy()

    if dropna_train:
        mask = ~X_train.isna().any(axis=1) & y_train.notna()
        X_train = X_train.loc[mask].copy()
        y_train = y_train.loc[mask].copy()

    X_forecast = forecast[feature_cols].copy()
    ids_forecast = forecast[["item_id", "store_id", "day_index"]].copy()

    return X_train, y_train, X_forecast, ids_forecast, feature_cols


# ----------------------- Recursive (primary mode) ------------------------


@dataclass(frozen=True)
class RecursiveFeatureConfig:
    """
    One-step-ahead training features (used for recursive forecasting).

    We train y[t] from features computed using only history < t.
    """
    lag_days: Tuple[int, ...] = (1, 7, 28)
    roll_windows: Tuple[int, ...] = (7, 28)

    # Price features (known)
    price_lag_days: Tuple[int, ...] = (7, 28)
    price_roll_windows: Tuple[int, ...] = (28,)

    include_event_type: bool = True


def make_one_step_train_matrix(
    df: pd.DataFrame,
    cutoff_day: int,
    cfg: RecursiveFeatureConfig,
    *,
    dropna_train: bool = True,
    train_window_days: int = 365,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build (X_train, y_train, feature_cols) for one-step-ahead training.

    Rows are days t within [cutoff_day-train_window_days+1, cutoff_day].
    Target is sales[t].
    Features:
      - sales lags: sales[t-l]
      - sales rolling: rolling on sales shifted(1) so window excludes current day
      - price lags/deltas: sell_price[t-l], sell_price[t] / sell_price[t-l] - 1
      - price rolling: rolling on price shifted(1)
      - exogenous calendar/event/snap (known)

    Leakage-safe for one-step training.
    """
    required = {
        "item_id", "store_id", "day_index", "sales",
        "dept_id", "cat_id", "state_id",
        "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1",
        "snap_CA", "snap_TX", "snap_WI",
        "sell_price",
    }
    _require_columns(df, required)

    min_day = max(1, cutoff_day - train_window_days + 1)

    out = df[(df["day_index"] >= min_day) & (df["day_index"] <= cutoff_day)].copy()
    out = out.sort_values(["item_id", "store_id", "day_index"]).reset_index(drop=True)

    # Exog + categoricals
    out = _add_common_exog(out)
    out = _cast_categoricals(out, ["item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday", "event_type_1"])

    g = out.groupby(["item_id", "store_id"], sort=False)

    # Sales lags
    for lag in cfg.lag_days:
        out[f"lag_{lag}"] = g["sales"].shift(lag)

    # Sales rolling (exclude current day by shifting 1)
    for w in cfg.roll_windows:
        out[f"roll_mean_{w}"] = g["sales"].transform(lambda s: s.shift(1).rolling(w, min_periods=w).mean())
        out[f"roll_std_{w}"] = g["sales"].transform(lambda s: s.shift(1).rolling(w, min_periods=w).std())

    # Price lags and deltas (known)
    for pl in cfg.price_lag_days:
        out[f"price_lag_{pl}"] = g["sell_price"].shift(pl)
        out[f"price_change_{pl}"] = (out["sell_price"] / out[f"price_lag_{pl}"]) - 1.0

    for w in cfg.price_roll_windows:
        out[f"price_roll_mean_{w}"] = g["sell_price"].transform(lambda s: s.shift(1).rolling(w, min_periods=w).mean())
        out[f"price_roll_std_{w}"] = g["sell_price"].transform(lambda s: s.shift(1).rolling(w, min_periods=w).std())

    # Feature columns
    feature_cols = [
        # IDs
        "item_id", "dept_id", "cat_id", "store_id", "state_id",
        # calendar
        "wday", "month", "year", "is_weekend", "weekday",
        # SNAP / events
        "snap", "has_event",
        # price base
        "sell_price",
    ]
    if cfg.include_event_type:
        feature_cols.append("event_type_1")

    feature_cols += [f"lag_{l}" for l in cfg.lag_days]
    for w in cfg.roll_windows:
        feature_cols += [f"roll_mean_{w}", f"roll_std_{w}"]

    for pl in cfg.price_lag_days:
        feature_cols += [f"price_lag_{pl}", f"price_change_{pl}"]
    for w in cfg.price_roll_windows:
        feature_cols += [f"price_roll_mean_{w}", f"price_roll_std_{w}"]

    X_train = out[feature_cols].copy()
    y_train = out["sales"].copy()

    if dropna_train:
        mask = ~X_train.isna().any(axis=1) & y_train.notna()
        X_train = X_train.loc[mask].copy()
        y_train = y_train.loc[mask].copy()

    return X_train, y_train, feature_cols
