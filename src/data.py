# scripts/prepare_data.py
# Goal: expose your data-prep as a CLI command using tyro, so filepath + store filters are provided at runtime.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro


def load_raw_data(data_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw M5 CSV files from a directory."""
    data_path = Path(data_path)

    sales_wide = pd.read_csv(data_path / "sales_train_evaluation.csv")
    calendar = pd.read_csv(data_path / "calendar.csv")
    prices = pd.read_csv(data_path / "sell_prices.csv")

    return sales_wide, calendar, prices


def melt_sales(sales_wide: pd.DataFrame) -> pd.DataFrame:
    """Convert sales from wide format to long format."""
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_columns = [col for col in sales_wide.columns if col.startswith("d_")]

    return pd.melt(
        sales_wide,
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )


def merge_calendar(sales_long: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features to sales data."""
    calendar_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI",
    ]
    merged = sales_long.merge(calendar[calendar_cols].copy(), on="d", how="left")
    merged["date"] = pd.to_datetime(merged["date"])
    return merged


def merge_prices(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Add price information to sales data."""
    return df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")


def add_day_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric day index from 'd' column."""
    df = df.copy()
    df["day_index"] = df["d"].str.replace("d_", "").astype(int)
    return df


def filter_stores(df: pd.DataFrame, stores: list[str]) -> pd.DataFrame:
    """Filter to only the specified stores."""
    return df[df["store_id"].isin(stores)].copy()


def load_and_prepare_data(data_path: str | Path, stores: Optional[list[str]] = None) -> pd.DataFrame:
    """Load raw data and prepare for modeling."""
    sales_wide, calendar, prices = load_raw_data(data_path)
    sales_long = melt_sales(sales_wide)

    if stores is not None:
        sales_long = filter_stores(sales_long, stores)

    df = merge_calendar(sales_long, calendar)
    df = merge_prices(df, prices)
    df = add_day_index(df)
    df = df.sort_values(["item_id", "store_id", "day_index"]).reset_index(drop=True)
    return df


@dataclass
class Args:
    # Directory containing the raw CSVs
    data_path: Path

    # Optional store filter: pass one or more store_ids like CA_1 CA_2 ...
    stores: Optional[list[str]] = None

    # Optional: if provided, write output here (csv or parquet)
    output_path: Optional[Path] = None


def main(args: Args) -> None:
    df = load_and_prepare_data(args.data_path, stores=args.stores)

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(df.head(10))
    print(df["sales"].describe())

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.output_path.suffix.lower() == ".parquet":
            df.to_parquet(args.output_path, index=False)
        else:
            df.to_csv(args.output_path, index=False)
        print(f"\nWrote: {args.output_path.resolve()}")


if __name__ == "__main__":
    main(tyro.cli(Args))
