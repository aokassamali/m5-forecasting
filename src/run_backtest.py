"""
Backtest orchestrator for M5 forecasting.

Usage examples:

# Naive baseline
python .\src\run_backtest.py --model naive --cutoffs 1857 1885 1913 --save-predictions

# LGBM direct (cutoff-anchored)
python .\src\run_backtest.py --model lgbm --cutoffs 1913 --lgbm-recursive False

# LGBM recursive (recommended) with train window
python .\src\run_backtest.py --model lgbm --cutoffs 1913 --lgbm-recursive --train-window-days 365 --lgbm-cfg.objective regression

Outputs:
- results/metrics/backtest_{model}_{run_tag}.csv
- results/summary/backtest_{model}_{run_tag}_summary.json
- (optional) results/predictions_df/{model}_cutoff_{cutoff}_{run_tag}.parquet
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import tyro

from evaluation import evaluate_predictions
from models.naive import seasonal_naive_predict
from models.lgbm import lgbm_forecast, LGBMConfig
from features import FeatureConfig


def ensure_actuals(preds: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure predictions have y_actual column by joining to df.sales.
    """
    if "y_actual" in preds.columns:
        return preds

    actuals = df[["item_id", "store_id", "day_index", "sales"]].copy()
    actuals = actuals.rename(columns={"sales": "y_actual"})
    out = preds.merge(actuals, on=["item_id", "store_id", "day_index"], how="left")
    return out


def _forecast(
    df: pd.DataFrame,
    model: Literal["naive", "lgbm"],
    cutoff_day: int,
    horizon: int,
    lag: int,
    feat_cfg: FeatureConfig,
    lgbm_cfg: LGBMConfig,
    lgbm_recursive: bool,
    train_window_days: int,
) -> pd.DataFrame:
    """
    Model dispatch. Returns predictions df: [item_id, store_id, day_index, y_pred].
    """
    if model == "naive":
        return seasonal_naive_predict(df, cutoff_day=cutoff_day, horizon=horizon, lag=lag)

    if model == "lgbm":
        # Keep feat_cfg horizon synced for direct mode (recursive mode ignores feat_cfg)
        feat_cfg = FeatureConfig(
            horizon=horizon,
            lag_days=feat_cfg.lag_days,
            roll_windows=feat_cfg.roll_windows,
            price_lag_days=feat_cfg.price_lag_days,
            price_roll_windows=feat_cfg.price_roll_windows,
            include_event_type=feat_cfg.include_event_type,
        )
        return lgbm_forecast(
            df,
            cutoff_day=cutoff_day,
            horizon=horizon,
            feat_cfg=feat_cfg,
            model_cfg=lgbm_cfg,
            train_window_days=train_window_days,
        )


    raise ValueError(f"Unknown model: {model}")


def run_single_cutoff(
    df: pd.DataFrame,
    model: Literal["naive", "lgbm"],
    cutoff_day: int,
    horizon: int,
    lag: int,
    feat_cfg: FeatureConfig,
    lgbm_cfg: LGBMConfig,
    lgbm_recursive: bool,
    train_window_days: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Run one cutoff: predict -> attach actuals -> evaluate.
    Returns (preds_with_actuals, metrics_dict).
    """
    preds = _forecast(
        df=df,
        model=model,
        cutoff_day=cutoff_day,
        horizon=horizon,
        lag=lag,
        feat_cfg=feat_cfg,
        lgbm_cfg=lgbm_cfg,
        lgbm_recursive=lgbm_recursive,
        train_window_days=train_window_days,
    )

    preds = ensure_actuals(preds, df)

    # Safety: evaluate only the forecast window
    preds = preds[(preds["day_index"] > cutoff_day) & (preds["day_index"] <= cutoff_day + horizon)].copy()

    metrics = evaluate_predictions(preds, df, cutoff_day=cutoff_day)
    metrics.update({"model": model, "horizon": horizon, "lag": lag})
    return preds, metrics


def run_backtest(
    df: pd.DataFrame,
    model: Literal["naive", "lgbm"],
    cutoffs: list[int],
    horizon: int,
    lag: int,
    feat_cfg: FeatureConfig,
    lgbm_cfg: LGBMConfig,
    lgbm_recursive: bool,
    train_window_days: int,
    *,
    save_predictions: bool,
    predictions_dir: Path,
    run_tag: str,
) -> pd.DataFrame:
    """
    Run multiple cutoffs and optionally save prediction artifacts.
    Returns DataFrame with one row per cutoff.
    """
    all_metrics: list[dict] = []

    if save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    for cutoff in cutoffs:
        print(f"Running {model} at cutoff {cutoff}...")
        preds, metrics = run_single_cutoff(
            df=df,
            model=model,
            cutoff_day=cutoff,
            horizon=horizon,
            lag=lag,
            feat_cfg=feat_cfg,
            lgbm_cfg=lgbm_cfg,
            lgbm_recursive=lgbm_recursive,
            train_window_days=train_window_days,
        )
        all_metrics.append(metrics)

        if save_predictions:
            pred_path = predictions_dir / f"{model}_cutoff_{cutoff}_{run_tag}.parquet"
            preds.to_parquet(pred_path, index=False)
            print(f"  Saved predictions -> {pred_path}")

    return pd.DataFrame(all_metrics)


@dataclass
class Args:
    # Data
    data_path: Path = Path("data/processed/m5_melted_merged.parquet")
    stores: Optional[list[str]] = None

    # Backtest config
    model: Literal["naive", "lgbm"] = "lgbm"
    cutoffs: list[int] = field(default_factory=lambda: [1857, 1885, 1913])
    horizon: int = 28
    lag: int = 7  # only used for naive

    # LGBM mode toggles
    lgbm_recursive: bool = True
    train_window_days: int = 365

    # Nested configs (tyro exposes these as flags)
    feat_cfg: FeatureConfig = field(default_factory=FeatureConfig)
    lgbm_cfg: LGBMConfig = field(default_factory=LGBMConfig)

    # Outputs
    results_metrics_dir: Path = Path("results/metrics")
    results_summary_dir: Path = Path("results/summary")
    predictions_dir: Path = Path("results/predictions_df")
    save_predictions: bool = False


def main(args: Args) -> None:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)

    # Optional store filter
    if args.stores is not None:
        df = df[df["store_id"].isin(args.stores)].copy()
        print(f"Filtered to stores: {args.stores}")

    print(f"Data shape: {df.shape}")
    print(f"Model: {args.model} | Cutoffs: {args.cutoffs} | Horizon: {args.horizon}")

    # Ensure output dirs
    args.results_metrics_dir.mkdir(parents=True, exist_ok=True)
    args.results_summary_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        args.predictions_dir.mkdir(parents=True, exist_ok=True)

    # Run backtest
    results_df = run_backtest(
        df=df,
        model=args.model,
        cutoffs=args.cutoffs,
        horizon=args.horizon,
        lag=args.lag,
        feat_cfg=args.feat_cfg,
        lgbm_cfg=args.lgbm_cfg,
        lgbm_recursive=args.lgbm_recursive,
        train_window_days=args.train_window_days,
        save_predictions=args.save_predictions,
        predictions_dir=args.predictions_dir,
        run_tag=run_tag,
    )

    # Save detailed metrics CSV (one row per cutoff)
    csv_path = args.results_metrics_dir / f"backtest_{args.model}_{run_tag}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics CSV -> {csv_path}")

    # Summary JSON
    summary = {
        "model": args.model,
        "cutoffs": args.cutoffs,
        "horizon": args.horizon,
        "lag": args.lag,
        "stores": args.stores,
        "lgbm_recursive": args.lgbm_recursive,
        "train_window_days": args.train_window_days,
        "timestamp": run_tag,
        "n_cutoffs": len(args.cutoffs),
        "mae_mean": float(results_df["mae"].mean()),
        "mae_std": float(results_df["mae"].std()) if len(results_df) > 1 else None,
        "rmse_mean": float(results_df["rmse"].mean()),
        "rmse_std": float(results_df["rmse"].std()) if len(results_df) > 1 else None,
        "wrmsse_mean": float(results_df["wrmsse"].mean()),
        "wrmsse_std": float(results_df["wrmsse"].std()) if len(results_df) > 1 else None,
    }

    summary_path = args.results_summary_dir / f"backtest_{args.model}_{run_tag}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON -> {summary_path}")

    # Print table
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    cols = ["cutoff_day", "mae", "rmse", "wrmsse", "n_valid"]
    print(results_df[cols].to_string(index=False))
    print("-" * 60)
    print(f"Mean WRMSSE: {summary['wrmsse_mean']:.4f} (+/- {summary['wrmsse_std']})")
    print("=" * 60)


if __name__ == "__main__":
    main(tyro.cli(Args))
