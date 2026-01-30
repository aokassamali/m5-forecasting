# Save this as debug_features.py and run it
import pandas as pd
from features import FeatureConfig, build_feature_frame_direct

df = pd.read_parquet("data/processed/m5_melted_merged.parquet")

# Use small subset for speed
df_small = df[df["store_id"] == "CA_1"].copy()

cfg = FeatureConfig(horizon=28)
feat_df, feature_cols = build_feature_frame_direct(df_small, cutoff_day=1913, cfg=cfg)

# Check forecast rows
forecast = feat_df[feat_df["day_index"] > 1913].copy()

print("=== FORECAST ROWS ===")
print(f"Shape: {forecast.shape}")
print(f"\nNaN counts per feature:")
print(forecast[feature_cols].isna().sum())
print(f"\nSample forecast row features:")
print(forecast[feature_cols].head(1).T)