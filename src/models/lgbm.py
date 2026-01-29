from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from features import FeatureConfig, build_feature_frame_direct


# ----------------------------
# Config
# ----------------------------
@dataclass
class LGBMConfig:
    # Core objective for the conditional-mean regressor (used in hurdle regressor on y>0)
    objective: str = "regression"  # "regression" or "poisson" recommended here

    # Tree params
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 400
    min_child_samples: int = 50

    # Sampling / regularization
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0

    # Misc
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = -1

    # Hurdle
    hurdle: bool = True

    # Optional (you said you added this already; leaving it here for completeness)
    tweedie_variance_power: float = 1.1

    # Experts / routing
    use_experts: bool = False
    sparse_tier_name: str = "High"
    sparse_p_threshold: float = 0.0  # set to 0.5 to gate sparse positives


# ----------------------------
# Internal helpers
# ----------------------------
def _categorical_features(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if str(X[c].dtype) == "category"]

def _ensure_categoricals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns and str(df[c].dtype) != "category":
            df[c] = df[c].astype("category")
    return df



def _build_classifier(cfg: LGBMConfig) -> lgb.LGBMClassifier:
    # Goal: good calibration of P(y>0)
    return lgb.LGBMClassifier(
        objective="binary",
        num_leaves=cfg.num_leaves,
        learning_rate=cfg.learning_rate,
        n_estimators=max(200, cfg.n_estimators // 2),
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        verbose=cfg.verbosity,
    )


def _build_regressor(cfg: LGBMConfig) -> lgb.LGBMRegressor:
    # Note: we are intentionally NOT using tweedie here because it performed catastrophically
    # in your current setup. Keep objective in {"regression","poisson"} for now.
    return lgb.LGBMRegressor(
        objective=cfg.objective,
        num_leaves=cfg.num_leaves,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        verbose=cfg.verbosity,
    )


def _compute_intermittency_tiers(
    df: pd.DataFrame,
    cutoff_day: int,
    n_tiers: int = 3,
) -> pd.DataFrame:
    """
    Training-only intermittency tiers by % zero-sales days.
    Matches your slicing logic: qcut over per-series zero_pct.
    """
    train = df[df["day_index"] <= cutoff_day][["item_id", "store_id", "sales"]].copy()
    grp = train.groupby(["item_id", "store_id"], sort=False)["sales"]

    # % of days with zero sales
    zero_pct = grp.apply(lambda s: float((s == 0).mean())).reset_index(name="zero_pct")

    labels = ["Low", "Medium", "High"][:n_tiers]
    zero_pct["intermittency_tier"] = pd.qcut(
        zero_pct["zero_pct"],
        q=n_tiers,
        labels=labels,
        duplicates="drop",
    )

    return zero_pct[["item_id", "store_id", "zero_pct", "intermittency_tier"]]


def _fit_hurdle(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: LGBMConfig,
) -> tuple[lgb.LGBMClassifier, lgb.LGBMRegressor]:
    """
    Fit:
      - classifier on y>0
      - regressor on y (only where y>0)
    """
    # Force known ID columns to category if present (LightGBM requires this)
    force_cat = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday", "event_type_1"]
    X_train = X_train.copy()
    for c in force_cat:
        if c in X_train.columns and str(X_train[c].dtype) != "category":
            X_train[c] = X_train[c].astype("category")

    clf = _build_classifier(cfg)
    reg = _build_regressor(cfg)

    cat_cols = _categorical_features(X_train)

    y_bin = (y_train > 0).astype(int)
    clf.fit(X_train, y_bin, categorical_feature=cat_cols)

    pos_mask = y_train > 0
    if int(pos_mask.sum()) == 0:
        # Degenerate: no positive examples
        # Still return fitted classifier; regressor will never be used meaningfully
        reg.fit(X_train.iloc[:1], y_train.iloc[:1], categorical_feature=cat_cols)
        return clf, reg

    reg.fit(X_train.loc[pos_mask], y_train.loc[pos_mask], categorical_feature=cat_cols)
    return clf, reg


def _predict_hurdle(
    clf: lgb.LGBMClassifier,
    reg: lgb.LGBMRegressor,
    X_forecast: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (p_positive, mu_positive).
    """
    # Force categoricals
    force_cat = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday", "event_type_1"]
    X_forecast = X_forecast.copy()
    for c in force_cat:
        if c in X_forecast.columns and str(X_forecast[c].dtype) != "category":
            X_forecast[c] = X_forecast[c].astype("category")

    # P(y>0)
    p = clf.predict_proba(X_forecast)[:, 1]

    # E[y | y>0] (regressor trained only on positives)
    mu = reg.predict(X_forecast)

    # Safety: enforce nonnegativity
    mu = np.maximum(mu, 0.0)
    print("p stats:", np.quantile(p, [0, .5, .9, .99, 1]))
    print("mu stats:", np.quantile(mu, [0, .5, .9, .99, 1]))

    return p, mu


# ----------------------------
# Public API used by run_backtest
# ----------------------------
def lgbm_forecast(
    df: pd.DataFrame,
    cutoff_day: int,
    horizon: int,
    *,
    feat_cfg: FeatureConfig,
    model_cfg: LGBMConfig,
    dropna_train: bool = True,
    train_window_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Train (one or two) hurdle LightGBM models and forecast cutoff+1..cutoff+horizon.

    Returns:
        DataFrame with columns [item_id, store_id, day_index, y_pred]
    """
    # Force feature horizon to match backtest horizon
    print("LGBMConfig:", model_cfg)

    feat_cfg = FeatureConfig(
        horizon=horizon,
        lag_days=feat_cfg.lag_days,
        roll_windows=feat_cfg.roll_windows,
        price_lag_days=feat_cfg.price_lag_days,
        price_roll_windows=feat_cfg.price_roll_windows,
        include_event_type=feat_cfg.include_event_type,
        include_intermittency_state=getattr(feat_cfg, "include_intermittency_state", True),
        days_since_nonzero_cap=getattr(feat_cfg, "days_since_nonzero_cap", 999),
    )

    feat_df, feature_cols = build_feature_frame_direct(df, cutoff_day, feat_cfg)

    # Train / forecast split (what you pasted)
    train = feat_df[feat_df["day_index"] <= cutoff_day].copy()
    forecast = feat_df[
        (feat_df["day_index"] > cutoff_day) & (feat_df["day_index"] <= cutoff_day + horizon)
    ].copy()

    # Ensure LightGBM can accept ID columns as categoricals
    cat_force = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday", "event_type_1"]
    train = _ensure_categoricals(train, cat_force)
    forecast = _ensure_categoricals(forecast, cat_force)


    # Optional rolling train window
    if train_window_days is not None:
        lo = cutoff_day - int(train_window_days) + 1
        train = train[train["day_index"] >= lo].copy()

    # Drop NA rows (mostly from lags/rolls)
    if dropna_train:
        train = train.dropna(subset=feature_cols + ["sales"]).copy()

    # Matrices
    X_train = train[feature_cols]
    y_train = train["sales"].astype(float)

    X_forecast = forecast[feature_cols]
    ids_forecast = forecast[["item_id", "store_id", "day_index"]].copy()

    # Expert routing label (per series, training only)
    if model_cfg.use_experts:
        tiers = _compute_intermittency_tiers(df, cutoff_day=cutoff_day)
        # Attach tier to both train and forecast rows
        train = train.merge(tiers[["item_id", "store_id", "intermittency_tier"]],
                            on=["item_id", "store_id"], how="left")
        forecast = forecast.merge(tiers[["item_id", "store_id", "intermittency_tier"]],
                                  on=["item_id", "store_id"], how="left")

        sparse_name = model_cfg.sparse_tier_name
        is_sparse_train = (train["intermittency_tier"] == sparse_name).fillna(False)
        is_sparse_forecast = (forecast["intermittency_tier"] == sparse_name).fillna(False)

        # Dense expert: everything NOT sparse
        dense_train = train.loc[~is_sparse_train].copy()
        sparse_train = train.loc[is_sparse_train].copy()

        # If a split is empty, fall back to global training for that expert
        if len(dense_train) == 0:
            dense_train = train.copy()
        if len(sparse_train) == 0:
            sparse_train = train.copy()

        # Fit experts
        clf_dense, reg_dense = _fit_hurdle(dense_train[feature_cols], dense_train["sales"], model_cfg)
        clf_sparse, reg_sparse = _fit_hurdle(sparse_train[feature_cols], sparse_train["sales"], model_cfg)

        # Predict
        p_dense, mu_dense = _predict_hurdle(clf_dense, reg_dense, X_forecast)
        p_sparse, mu_sparse = _predict_hurdle(clf_sparse, reg_sparse, X_forecast)

        # Combine with routing
        y_pred = np.empty(len(forecast), dtype=float)

        # Default: dense prediction
        y_pred[:] = p_dense * mu_dense

        # Sparse routes
        sparse_idx = np.where(is_sparse_forecast.to_numpy())[0]
        if len(sparse_idx) > 0:
            p_s = p_sparse[sparse_idx]
            mu_s = mu_sparse[sparse_idx]

            # Apply sparse gating threshold
            thr = float(model_cfg.sparse_p_threshold)
            if thr > 0.0:
                gated = (p_s >= thr).astype(float)  # 1 if confident positive else 0
                y_pred[sparse_idx] = gated * (p_s * mu_s)
            else:
                y_pred[sparse_idx] = p_s * mu_s

    else:
        # Single global hurdle
        clf, reg = _fit_hurdle(X_train, y_train, model_cfg)
        p, mu = _predict_hurdle(clf, reg, X_forecast)
        y_pred = p * mu

    # Final safety
    y_pred = np.maximum(y_pred, 0.0)

    out = ids_forecast.copy()
    out["y_pred"] = y_pred.astype(float)
    return out
