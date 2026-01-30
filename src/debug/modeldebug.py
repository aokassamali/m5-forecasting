# Save as debug_lgbm.py and run it
import pandas as pd
import numpy as np
import lightgbm as lgb
from features import FeatureConfig, build_feature_frame_direct

df = pd.read_parquet("data/processed/m5_melted_merged.parquet")

# Use ONE store for speed
df = df[df["store_id"] == "CA_1"].copy()
print(f"Data shape: {df.shape}")

cfg = FeatureConfig(horizon=28)
feat_df, feature_cols = build_feature_frame_direct(df, cutoff_day=1913, cfg=cfg)

# Remove 'h' from features
feature_cols = [c for c in feature_cols if c != "h"]
# Remove year from feature_cols
feature_cols = [c for c in feature_cols if c != "year"]

# Split
train = feat_df[feat_df["day_index"] <= 1913].copy()
forecast = feat_df[feat_df["day_index"] > 1913].copy()

print(f"Train shape: {train.shape}")
print(f"Forecast shape: {forecast.shape}")

# Drop NA
train_clean = train.dropna(subset=feature_cols + ["sales"])
print(f"Train after dropna: {train_clean.shape}")

# Prepare X, y
X_train = train_clean[feature_cols]
y_train = train_clean["sales"]

print(f"\ny_train stats: min={y_train.min()}, mean={y_train.mean():.2f}, max={y_train.max()}")
print(f"y_train zeros: {(y_train == 0).sum()} / {len(y_train)}")

# Check categorical columns
cat_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday", "event_type_1"]
cat_cols = [c for c in cat_cols if c in feature_cols]

for c in cat_cols:
    X_train[c] = X_train[c].astype("category")

# Train simple model
print("\nTraining LightGBM...")
model = lgb.LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.05,
    verbose=-1,
)
model.fit(X_train, y_train, categorical_feature=cat_cols)

# Predict on TRAINING data first (sanity check)
train_preds = model.predict(X_train)
print(f"\nTrain predictions: min={train_preds.min():.2f}, mean={train_preds.mean():.2f}, max={train_preds.max():.2f}")

# Now predict on forecast
X_forecast = forecast[feature_cols].copy()
for c in cat_cols:
    X_forecast[c] = X_forecast[c].astype("category")

# Fill NaN in numeric columns only
numeric_cols = X_forecast.select_dtypes(include=[np.number]).columns
X_forecast[numeric_cols] = X_forecast[numeric_cols].fillna(0)

forecast_preds = model.predict(X_forecast)
print(f"Forecast predictions: min={forecast_preds.min():.2f}, mean={forecast_preds.mean():.2f}, max={forecast_preds.max():.2f}")

# Add this to the end of your debug script:

print("\n=== CHECKING CATEGORICAL ALIGNMENT ===")
for c in cat_cols:
    train_cats = set(X_train[c].cat.categories)
    forecast_cats = set(X_forecast[c].cat.categories)
    
    only_in_forecast = forecast_cats - train_cats
    if only_in_forecast:
        print(f"{c}: {len(only_in_forecast)} categories in forecast but not in train")
        print(f"   Examples: {list(only_in_forecast)[:5]}")
    else:
        print(f"{c}: OK")

print("\n=== CHECKING ITEM COVERAGE ===")
train_items = set(X_train["item_id"].unique())
forecast_items = set(X_forecast["item_id"].unique())

print(f"Items in train: {len(train_items)}")
print(f"Items in forecast: {len(forecast_items)}")
print(f"Items in forecast but NOT in train: {len(forecast_items - train_items)}")

print("\n=== COMPARING FEATURE DISTRIBUTIONS ===")
numeric_features = [c for c in feature_cols if c not in cat_cols]

for c in numeric_features:
    train_mean = X_train[c].mean()
    train_std = X_train[c].std()
    forecast_mean = X_forecast[c].mean()
    forecast_std = X_forecast[c].std()
    
    print(f"{c}:")
    print(f"   Train:    mean={train_mean:.3f}, std={train_std:.3f}")
    print(f"   Forecast: mean={forecast_mean:.3f}, std={forecast_std:.3f}")

# Check feature importances
print("\n=== FEATURE IMPORTANCES ===")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.head(15))

# Try predicting on a SINGLE forecast row with explicit values
print("\n=== SINGLE ROW TEST ===")
single_row = X_forecast.iloc[[0]].copy()
print("Input features:")
print(single_row.T)
pred = model.predict(single_row)
print(f"Prediction: {pred[0]}")