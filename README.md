# M5 Forecasting Harness

A production-quality demand forecasting system demonstrating rigorous evaluation methodology, honest baseline comparison, and interpretable error analysis on Walmart's M5 retail dataset.

## Key Results

| Model | WRMSSE | MAE | vs. Baseline |
|-------|--------|-----|--------------|
| Seasonal Naive (lag-7) | 1.32 ± 0.02 | 1.22 | — |
| **LightGBM** | **1.03 ± 0.02** | **1.07** | **-22%** |

LightGBM achieves a 22% reduction in WRMSSE compared to the seasonal naive baseline, with consistent performance across three rolling-origin evaluation windows.

---

## Business Context

### The Problem
Walmart operates 10 stores across 3 states (CA, TX, WI), selling 3,049 products across 7 departments. Accurate 28-day demand forecasts drive:

- **Inventory optimization**: Reduce stockouts (lost sales) and overstock (markdowns, waste)
- **Labor planning**: Staff scheduling based on expected demand
- **Supply chain coordination**: Warehouse allocation, transportation scheduling

A 22% improvement in forecast accuracy translates directly to reduced safety stock requirements and improved service levels.

### Why WRMSSE?
We use Weighted Root Mean Squared Scaled Error (WRMSSE) rather than simple MAE because:

1. **Scale-invariant**: An error of 5 units means different things for an item selling 100/day vs 1/day
2. **Revenue-weighted**: High-revenue items contribute more to the metric, aligning with business impact
3. **Benchmarked**: Errors are scaled relative to a naive lag-1 baseline (WRMSSE = 1.0 means "no better than yesterday's value")

---

## Methodology

### Evaluation Protocol
We use **rolling-origin backtesting** to simulate realistic forecast deployment:

```
Cutoff 1857 → Forecast days 1858-1885 (Feb 2016)
Cutoff 1885 → Forecast days 1886-1913 (Mar 2016)  
Cutoff 1913 → Forecast days 1914-1941 (Apr 2016)
```

Each cutoff trains only on historical data, forecasts 28 days ahead, and evaluates against actuals. Reporting mean ± std across cutoffs guards against overfitting to a single time period.

### Leakage Prevention
All features are constructed using only data available at forecast time:

- Lag features use minimum lag of 28 days (equal to forecast horizon)
- Rolling statistics shift by 28 days before computing windows
- Price features are assumed known (retailers set prices in advance)
- Calendar/SNAP features are deterministic

---

## Model Comparison

### Seasonal Naive Baseline
**Prediction**: Sales for day t = Sales from day t-7 (same weekday last week)

This captures weekly shopping patterns (weekend vs. weekday) with zero training required. It's the "beat this or your ML added no value" benchmark.

**Results**: WRMSSE = 1.32

Surprisingly, seasonal naive performs *worse* than the implicit lag-1 benchmark (WRMSSE > 1.0). Investigation via error slicing revealed why.

### LightGBM
**Features**:
- Lag features: lag_28, lag_35, lag_42, lag_49, lag_56
- Rolling statistics: 7-day and 28-day mean/std (shifted by horizon)
- Calendar: day of week, month, weekend flag
- SNAP: State-specific food stamp benefit days
- Price: Current price, 7-day price change

**Hyperparameters**:
- Objective: Tweedie (variance power 1.1) — handles zero-inflated count data
- Trees: 200 estimators, 255 leaves, learning rate 0.03
- Regularization: 80% row/column subsampling

**Results**: WRMSSE = 1.03 ± 0.02

---

## Error Analysis: Where Models Win and Lose

The aggregate WRMSSE hides critical heterogeneity. Slicing by item characteristics reveals actionable patterns.

### By Sales Volume

| Volume Tier | Seasonal Naive | LightGBM | Winner |
|-------------|----------------|----------|--------|
| High | 1.18 | **0.94** | LightGBM (-20%) |
| Medium | 1.39 | **1.04** | LightGBM (-25%) |
| Low | 1.80 | **1.34** | LightGBM (-26%) |

**Insight**: LightGBM improves across all volume tiers, with the largest relative gains on medium and low-volume items where seasonal naive struggles most.

### By Demand Intermittency

| Intermittency | Seasonal Naive | LightGBM | Winner |
|---------------|----------------|----------|--------|
| Low (consistent sales) | 1.10 | **0.87** | LightGBM (-21%) |
| Medium | 1.45 | **1.11** | LightGBM (-23%) |
| High (sporadic/zeros) | 1.95 | **1.48** | LightGBM (-24%) |

**Insight**: Both models struggle with highly intermittent items (WRMSSE > 1.4), but LightGBM consistently outperforms. The challenge is fundamental: when an item sells 0-3 times per week randomly, even sophisticated models have limited signal.

### Business Implication
For a production system, consider **tiered model deployment**:

- **High-volume, low-intermittency items**: LightGBM with tight prediction intervals → aggressive inventory optimization
- **Sparse/intermittent items**: LightGBM with wider safety stock → prioritize service level over efficiency

---

## Reproducing Results

### Setup
```bash
# Clone and install
git clone <repo>
cd m5-forecasting
pip install -e .

# Download M5 data from Kaggle
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip -d data/raw/

# Prepare data
python src/data.py
```

### Run Backtests
```bash
# Seasonal naive baseline
python src/run_backtest.py --model naive --cutoffs 1857 1885 1913

# LightGBM
python src/run_backtest.py --model lgbm --cutoffs 1857 1885 1913

# Sliced evaluation
python scripts/run_sliced_eval.py --model lgbm --cutoff-day 1913
```

### Output
Results are saved to:
- `results/metrics/backtest_{model}_{timestamp}.csv` — per-cutoff metrics
- `results/summary/backtest_{model}_{timestamp}_summary.json` — aggregate statistics

---

## Repository Structure

```
m5-forecasting/
├── configs/
│   └── main.yaml                 # Cutoffs, horizons, hyperparameters
├── data/
│   ├── raw/                      # Original M5 CSVs (gitignored)
│   └── processed/                # Prepared parquet
├── src/
│   ├── data.py                   # Data loading and preparation
│   ├── evaluation.py             # WRMSSE, MAE, RMSE metrics
│   ├── slicing.py                # Volume/intermittency tier analysis
│   ├── run_backtest.py           # Main orchestrator
│   └── models/
│       ├── naive.py              # Seasonal naive baseline
│       └── lgbm.py               # LightGBM forecaster
├── scripts/
│   └── run_sliced_eval.py        # Sliced evaluation analysis
├── results/
│   ├── metrics/                  # Backtest outputs
│   └── summary/                  # Aggregate statistics
└── reports/
    └── model_card_lgbm.md        # Model documentation
```

---

## Limitations and Future Work

### Current Limitations

1. **No hierarchical reconciliation**: Forecasts are made independently per item-store. In production, forecasts should be coherent across the hierarchy (item → department → store → total). Methods like MinTrace or bottom-up aggregation would ensure consistency.

2. **Point forecasts only**: We predict expected sales but don't quantify uncertainty. For inventory optimization, prediction intervals (e.g., 80% coverage) are essential to set safety stock levels.

3. **Static model**: The model is trained once per cutoff. In production, concept drift (changing consumer behavior, new products, seasonality shifts) requires continuous monitoring and retraining triggers.

4. **No external data**: We use only sales, calendar, and price. Real systems incorporate weather, promotions calendar, competitor actions, and macroeconomic indicators.

### Recommended Extensions

**For Production Deployment**:

1. **Prediction Intervals**: Train quantile regression models (10th/90th percentile) using LightGBM's `objective="quantile"`. Validate calibration: 80% of actuals should fall within bounds.

2. **Hierarchical Reconciliation**: Implement bottom-up or optimal combination (MinTrace) to ensure store-level forecasts sum correctly to regional and national totals.

3. **Monitoring Dashboard**: Track forecast accuracy in real-time. Alert on:
   - Rolling WRMSSE exceeding threshold (e.g., > 1.2)
   - Systematic bias (consistent over/under-prediction)
   - Feature drift (input distributions shifting)

4. **Automated Retraining**: Trigger model refresh when:
   - Accuracy degrades beyond threshold
   - New products launch (cold-start handling)
   - Major events occur (e.g., pandemic, store remodel)

**For Research**:

1. **Neural approaches**: N-BEATS, Temporal Fusion Transformer for capturing complex temporal dynamics
2. **Causal modeling**: Estimate true promotional lift vs. correlation
3. **Intermittent demand**: Croston's method or specialized zero-inflated models for sparse items

---

## What I Learned

This project reinforced several principles:

1. **Baselines matter**: The seasonal naive "underperformance" (WRMSSE > 1.0) was itself an insight—retail demand at the item-store level is noisier than expected. Starting with a simple baseline revealed this.

2. **Aggregate metrics hide heterogeneity**: Overall WRMSSE of 1.03 sounds good, but high-intermittency items still have WRMSSE of 1.48. A production system needs segment-specific strategies.

3. **Feature engineering > model complexity**: The winning approach used simple lag/rolling features with LightGBM. Attempts to add categorical embeddings (item_id, store_id) caused train/test distribution mismatches. Simpler was better.

4. **Evaluation protocol is everything**: Rolling-origin backtesting with multiple cutoffs prevented overfitting to a single time period. The low variance across cutoffs (±0.02) gives confidence in the results.

---

## References

- [M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy)
- [M5 Competitors Guide](https://mofc.unic.ac.cy/m5-competition/)
- [1st Place Solution Summary](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163684)
- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd edition. OTexts.

---

## License

MIT
