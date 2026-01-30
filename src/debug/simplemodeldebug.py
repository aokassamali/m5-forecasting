# Save as debug_per_store.py
import pandas as pd
import numpy as np
import lightgbm as lgb

df = pd.read_parquet("data/processed/m5_melted_merged.parquet")
df = df[df["store_id"] == "CA_1"].copy()  # Single store

cutoff = 1913
horizon = 28

# Sort
df = df.sort_values(["item_id", "day_index"]).reset_index(drop=True)

# Simple features
g = df.groupby("item_id", sort=False)
df["lag_28"] = g["sales"].shift(28)
df["lag_35"] = g["sales"].shift(35)
df["lag_42"] = g["sales"].shift(42)
df["roll_mean_7"] = g["sales"].transform(lambda x: x.shift(28).rolling(7, min_periods=1).mean())
df["roll_mean_28"] = g["sales"].transform(lambda x: x.shift(28).rolling(28, min_periods=1).mean())

# Calendar
df["wday"] = df["wday"].astype(int)
df["month"] = df["month"].astype(int)

# Split
train = df[df["day_index"] <= cutoff].copy()
forecast = df[(df["day_index"] > cutoff) & (df["day_index"] <= cutoff + horizon)].copy()

feature_cols = ["lag_28", "lag_35", "lag_42", "roll_mean_7", "roll_mean_28", "wday", "month", "sell_price"]

train_clean = train.dropna(subset=feature_cols + ["sales"])
print(f"Train: {len(train_clean):,}")

X_train = train_clean[feature_cols]
y_train = train_clean["sales"]

X_forecast = forecast[feature_cols].fillna(0)

# Train
model = lgb.LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.1,
    n_estimators=200,
    num_leaves=255,
    learning_rate=0.03,
    verbose=-1,
)
model.fit(X_train, y_train)

# Predict
train_pred = model.predict(X_train)
forecast_pred = model.predict(X_forecast)

print(f"y_train mean: {y_train.mean():.2f}")
print(f"Train pred mean: {train_pred.mean():.2f}")
print(f"Forecast pred mean: {forecast_pred.mean():.2f}")
print(f"Forecast pred min/max: {forecast_pred.min():.2f} / {forecast_pred.max():.2f}")