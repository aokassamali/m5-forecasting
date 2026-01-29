import pandas as pd

p = pd.read_parquet(r"results\predictions_df\lgbm_cutoff_1913_20260128_224141.parquet")
print(p["y_pred"].describe(percentiles=[.5,.9,.99,.999]))
