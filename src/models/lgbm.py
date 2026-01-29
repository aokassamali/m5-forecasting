from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import inspect
import numpy as np
import pandas as pd

import lightgbm as lgb


# -----------------------------
# Config
# -----------------------------

@dataclass
class LGBMConfig:
    """
    LightGBM config for M5.

    Notes:
    - hurdle=True trains a binary classifier for y>0 and a regressor on positive rows, then predicts y_hat = p*mu.
    - objective applies to the regressor ("regression", "poisson", etc.).
    """

    # Regressor objective
    objective: str = "poisson"

    # Core tree params
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 300
    min_child_samples: int = 20

    # Regularization / sampling
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0

    # Misc
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = -1

    # Hurdle model
    hurdle: bool = False

    # Classifier params (kept separate so you can tune quickly)
    clf_num_leaves: int = 31
    clf_learning_rate: float = 0.05
    clf_n_estimators: int = 200
    clf_min_child_samples: int = 20

    # Safety: clip predictions
    clip_negative_to_zero: bool = True

    tweedie_variance_power: float = 1.1


# -----------------------------
# Helpers
# -----------------------------

def _as_frame(X: Any, feature_names: Optional[list[str]] = None) -> pd.DataFrame | Any:
    """Ensure we keep pandas dtypes (esp. categoricals) if the upstream builder returns a DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X
    if feature_names is not None:
        return pd.DataFrame(X, columns=feature_names)
    return X


def _call_make_train_forecast_matrices(
    df: pd.DataFrame,
    cutoff_day: int,
    feat_cfg: Any,
    *,
    dropna_train: bool,
    train_window_days: Optional[int],
):
    """
    Calls src.features.make_train_forecast_matrices but only passes kwargs that exist,
    to avoid constant signature churn breaking everything.
    """
    from features import make_train_forecast_matrices  # local import to avoid import cycles

    fn = make_train_forecast_matrices
    sig = inspect.signature(fn)

    kwargs: dict[str, Any] = {}
    if "dropna_train" in sig.parameters:
        kwargs["dropna_train"] = dropna_train
    if "train_window_days" in sig.parameters:
        kwargs["train_window_days"] = train_window_days

    out = fn(df, cutoff_day, feat_cfg, **kwargs)

    # Expected (based on your earlier trace):
    #   X_train, y_train, X_forecast, ids_forecast, feature_cols
    # But we’ll tolerate missing feature_cols.
    if len(out) == 5:
        return out
    if len(out) == 4:
        X_train, y_train, X_forecast, ids_forecast = out
        return X_train, y_train, X_forecast, ids_forecast, None

    raise ValueError(
        "make_train_forecast_matrices returned unexpected number of outputs. "
        f"Expected 4 or 5, got {len(out)}."
    )


def _build_regressor(cfg: LGBMConfig) -> lgb.LGBMRegressor:
    params = dict(
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

    # Tweedie-specific parameter
    if cfg.objective == "tweedie":
        params["tweedie_variance_power"] = cfg.tweedie_variance_power

    return lgb.LGBMRegressor(**params)



def _build_classifier(cfg: LGBMConfig) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        num_leaves=cfg.clf_num_leaves,
        learning_rate=cfg.clf_learning_rate,
        n_estimators=cfg.clf_n_estimators,
        min_child_samples=cfg.clf_min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        verbose=cfg.verbosity,
    )


# -----------------------------
# Public API
# -----------------------------

def lgbm_forecast(
    df: pd.DataFrame,
    *,
    cutoff_day: int,
    horizon: int,
    recursive: bool,
    feat_cfg: Any,
    model_cfg: LGBMConfig,
    dropna_train: bool = True,
    train_window_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Train LightGBM (optionally hurdle) and forecast.

    Returns:
        DataFrame with columns [item_id, store_id, day_index, y_pred]
    """
    # If your FeatureConfig stores horizon, keep it consistent with the backtest horizon
    # (This is important if your feature builder uses horizon internally.)
    if hasattr(feat_cfg, "horizon") and getattr(feat_cfg, "horizon") != horizon:
        try:
            # dataclass-style replace
            feat_cfg = type(feat_cfg)(**{**feat_cfg.__dict__, "horizon": horizon})
        except Exception:
            # best effort — if it isn't a dataclass, just overwrite
            setattr(feat_cfg, "horizon", horizon)

    # Build train and forecast matrices from your feature pipeline
    X_train, y_train, X_forecast, ids_forecast, feature_cols = _call_make_train_forecast_matrices(
        df=df,
        cutoff_day=cutoff_day,
        feat_cfg=feat_cfg,
        dropna_train=dropna_train,
        train_window_days=train_window_days,
    )

    X_train = _as_frame(X_train, feature_cols)
    X_forecast = _as_frame(X_forecast, feature_cols)

    # We keep ids_forecast as the identity table for the forecast rows
    # It should contain: item_id, store_id, day_index
    if not isinstance(ids_forecast, pd.DataFrame):
        raise ValueError("ids_forecast must be a DataFrame with identifiers for the forecast rows.")

    # -----------------------------
    # Train
    # -----------------------------
    reg = _build_regressor(model_cfg)

    if model_cfg.hurdle:
        # Classifier on all train rows
        y_bin = (pd.Series(y_train) > 0).astype(int).to_numpy()
        clf = _build_classifier(model_cfg)
        clf.fit(X_train, y_bin)

        # Regressor trained only on positive rows
        pos_mask = y_bin == 1
        if pos_mask.sum() == 0:
            # Degenerate edge-case: no positive examples in training window.
            # Fall back to always predicting 0.
            y_pred = np.zeros(len(ids_forecast), dtype=float)
        else:
            reg.fit(X_train.loc[pos_mask] if isinstance(X_train, pd.DataFrame) else X_train[pos_mask], np.asarray(y_train)[pos_mask])

            p = clf.predict_proba(X_forecast)[:, 1]
            mu = reg.predict(X_forecast)

            y_pred = p * mu
    else:
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_forecast)

    # -----------------------------
    # Post-process
    # -----------------------------
    y_pred = np.asarray(y_pred, dtype=float)

    if model_cfg.clip_negative_to_zero:
        y_pred = np.clip(y_pred, 0.0, None)

    # NOTE: "recursive" is accepted for API hygiene.
    # With your current leakage-safe features (min lag >= horizon), recursion usually does not change inputs.
    # A real recursive pipeline requires lag features < horizon (e.g., lag_1..lag_7), which your current builder likely avoids.
    # We keep the flag so run_backtest can pass it, but forecasting here is "direct" over the horizon.
    _ = recursive  # intentionally unused

    out = ids_forecast.copy()
    out["y_pred"] = y_pred

    # Normalize output columns
    expected_cols = ["item_id", "store_id", "day_index", "y_pred"]
    missing = [c for c in expected_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Forecast output is missing columns: {missing}. Got columns: {list(out.columns)}")

    return out[expected_cols]
