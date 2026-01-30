"""
Run evaluation with error slicing.

Usage:
  # Load predictions from artifact
  python scripts/run_sliced_eval.py `
    --data-path data/processed/m5_melted_merged.parquet `
    --preds-path results/predictions_df/lgbm_cutoff_1913_20260128_194457.parquet `
    --cutoff-day 1913

  # Generate seasonal naive predictions on the fly
  python scripts/run_sliced_eval.py `
    --data-path data/processed/m5_melted_merged.parquet `
    --model naive `
    --cutoff-day 1913

  # Generate LGBM predictions on the fly (calls models.lgbm.lgbm_forecast)
  python scripts/run_sliced_eval.py `
    --data-path data/processed/m5_melted_merged.parquet `
    --model lgbm `
    --cutoff-day 1913 `
    --horizon 28

Notes:
- If you pass --preds-path, the script will load those predictions and ignore --model.
- For LGBM, this script imports models.lgbm and calls lgbm_forecast.
  It will only pass arguments that lgbm_forecast actually accepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Literal, Optional
import inspect

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

    # If preds_path is None, we can generate predictions on the fly
    model: Literal["naive", "lgbm"] = "naive"
    horizon: int = 28
    lag: int = 7

    # Optional store filter
    stores: Optional[list[str]] = None

    # Optional LGBM knobs (only passed if lgbm_forecast accepts them)
    lgbm_model_path: Optional[Path] = None
    lgbm_run_tag: Optional[str] = None


def ensure_actuals(preds: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if "y_actual" in preds.columns:
        return preds
    actuals = df[["item_id", "store_id", "day_index", "sales"]].copy()
    actuals = actuals.rename(columns={"sales": "y_actual"})
    return preds.merge(actuals, on=["item_id", "store_id", "day_index"], how="left")


def _normalize_pred_col(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the predictions dataframe has a 'y_pred' column.
    Rename common alternatives to 'y_pred' when present.
    """
    if "y_pred" in preds.columns:
        return preds

    for cand in ["pred", "yhat", "forecast", "prediction", "sales_pred", "y_pred_lgbm"]:
        if cand in preds.columns:
            return preds.rename(columns={cand: "y_pred"})

    raise ValueError(
        "Predictions dataframe must contain a 'y_pred' column (or one of: "
        "pred, yhat, forecast, prediction, sales_pred, y_pred_lgbm). "
        f"Found columns: {list(preds.columns)}"
    )


def _call_lgbm_forecast(df: pd.DataFrame, args: Args) -> pd.DataFrame:
    """
    Import models.lgbm.lgbm_forecast and call it with a signature-aware kwargs filter.

    Expected output: a dataframe with columns:
      - item_id, store_id, day_index
      - y_pred (or a common alternative that we normalize)
    """
    mod = import_module("models.lgbm")
    if not hasattr(mod, "lgbm_forecast"):
        raise AttributeError("models.lgbm does not have 'lgbm_forecast'")

    fn = getattr(mod, "lgbm_forecast")

    # Candidate kwargs (we'll filter these against the real signature)
    candidate_kwargs = {
        "df": df,
        "cutoff_day": args.cutoff_day,
        "horizon": args.horizon,
        "model_path": args.lgbm_model_path,
        "lgbm_model_path": args.lgbm_model_path,  # in case your fn uses this name
        "run_tag": args.lgbm_run_tag,
        "lgbm_run_tag": args.lgbm_run_tag,        # in case your fn uses this name
    }

    # Drop Nones
    candidate_kwargs = {k: v for k, v in candidate_kwargs.items() if v is not None}

    sig = inspect.signature(fn)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    if not accepts_kwargs:
        candidate_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}

    preds = fn(**candidate_kwargs)
    if not isinstance(preds, pd.DataFrame):
        raise TypeError(f"lgbm_forecast must return a pandas DataFrame, got: {type(preds)}")

    preds = _normalize_pred_col(preds)

    required = {"item_id", "store_id", "day_index", "y_pred"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(
            f"lgbm_forecast output missing required columns: {sorted(missing)}. "
            f"Found: {list(preds.columns)}"
        )

    return preds


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
        preds = _normalize_pred_col(preds)
    else:
        print(f"Generating predictions with {args.model}...")
        if args.model == "naive":
            preds = seasonal_naive_predict(
                df, cutoff_day=args.cutoff_day, horizon=args.horizon, lag=args.lag
            )
            preds = _normalize_pred_col(preds)
        elif args.model == "lgbm":
            preds = _call_lgbm_forecast(df, args)
        else:
            raise ValueError(f"Unsupported model: {args.model}")

    preds = ensure_actuals(preds, df)

    # Keep only forecast window
    preds = preds[
        (preds["day_index"] > args.cutoff_day)
        & (preds["day_index"] <= args.cutoff_day + args.horizon)
    ].copy()

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
