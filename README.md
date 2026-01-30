# M5 Forecasting Harness

An evaluation methodology, honest baseline comparison, and interpretable error analysis on Walmart's M5 retail dataset.

> **Important:** The headline metric reported below is a **bottom-level WRMSSE proxy** (item × store only) intended for honest internal comparison across models in this repo.  
> It is **not directly comparable** to Kaggle leaderboard WRMSSE for the official M5 Accuracy competition (which includes 12 hierarchical levels and an “active period” scaling rule).

---

## Key Results

| Model | WRMSSE (proxy) | MAE | vs. Baseline |
|-------|-----------------|-----|--------------|
| Seasonal Naive (lag-7) | 1.32 ± 0.02 | 1.22 | — |
| **LightGBM** | **1.03 ± 0.02** | **1.07** | **-22%** |

LightGBM achieves a ~22% reduction in this repo’s WRMSSE proxy compared to the seasonal naive baseline, with consistent performance across three rolling-origin evaluation windows.

---

## Business Context

### The Problem
Walmart operates 10 stores across 3 states (CA, TX, WI), selling 3,049 products across 7 departments. Accurate 28-day demand forecasts drive:

- **Inventory optimization**: Reduce stockouts (lost sales) and overstock (markdowns, waste)
- **Labor planning**: Staff scheduling based on expected demand
- **Supply chain coordination**: Warehouse allocation, transportation scheduling

A 22% improvement in forecast accuracy translates directly to reduced safety stock requirements and improved service levels—especially if improvements concentrate on the highest-revenue series.

### Why WRMSSE (and why I call it a “proxy” here)
WRMSSE (Weighted Root Mean Squared Scaled Error) is useful because it is:

1. **Scale-invariant**: An error of 5 units means different things for an item selling 100/day vs 1/day  
2. **Business-weighted**: High-revenue items contribute more, aligning the score with impact  
3. **Comparable across series**: Errors are scaled by each series’ historical “typical” day-to-day variation (via squared first differences)

**Interpreting the value:** Lower is better. In this repo’s implementation, a value near **1.0** roughly means “forecast RMSE is on the order of the series’ typical day-to-day change in training.” It is **not** a literal statement about a lag-1 forecasting baseline unless you explicitly compute and compare to a lag-1 model.

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

> Note: The current forecasting mode is **one-step recursive**. This is simple to implement but can accumulate error across the 28-day rollout if features are not updated perfectly.

---

## Metric Details (What this repo computes)

This repo computes a **bottom-level WRMSSE proxy** for item-store series:

- **Scale per series**:  
  ![scale](https://latex.codecogs.com/png.latex?%5Ctext%7Bscale%7D_i%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7D%7Bn_i-1%7D%5Csum_%7Bt%3D2%7D%5E%7Bn_i%7D%28y_t-y_%7Bt-1%7D%29%5E2%7D) 
- **RMSSE per series** over the 28-day horizon:  
  ![rmsse](https://latex.codecogs.com/png.latex?%5Cmathrm%7BRMSSE%7D_i%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7Dh%5Csum_%7Bt%3D1%7D%5Eh%5Cleft%28%5Cfrac%7By_%7Bn%2Bt%7D-%5Chat%7By%7D_%7Bn%2Bt%7D%7D%7B%5Ctext%7Bscale%7D_i%7D%5Cright%29%5E2%7D)
- **Weights**: revenue share over the last 28 training days (units × price), normalized to sum to 1 across item-store series.

### How this differs from official Kaggle M5 WRMSSE
If you want a leaderboard-comparable WRMSSE, the official competition metric differs in two big ways:

1. **Hierarchy:** official M5 evaluates **12 aggregation levels** (42,840 series), not just item-store.  
2. **Scaling rule:** official M5 computes the scaling term after the series becomes “active” (after the first non-zero demand), whereas this repo’s current scale uses the full training window.

This repo is explicit about this because the goal here is **honest, reproducible model comparison** and **clear error analysis**, not leaderboard replication.

---

## Model Comparison

### Seasonal Naive Baseline
**Prediction**: Sales for day t = Sales from day t-7 (same weekday last week)

This captures weekly shopping patterns (weekend vs. weekday) with zero training required. If we can't beat a naive model, no point introducing complexity into a forecasting system.

**Results**: WRMSSE (proxy) = 1.32

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

**Results**: WRMSSE (proxy) = 1.03 ± 0.02

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

> Recommendation: also slice by **weight decile** (revenue share). Because WRMSSE is weighted, the “top decile” often dominates the overall score.

### By Demand Intermittency

| Intermittency | Seasonal Naive | LightGBM | Winner |
|---------------|----------------|----------|--------|
| Low (consistent sales) | 1.10 | **0.87** | LightGBM (-21%) |
| Medium | 1.45 | **1.11** | LightGBM (-23%) |
| High (sporadic/zeros) | 1.95 | **1.48** | LightGBM (-24%) |

**Insight**: Both models struggle with highly intermittent items (WRMSSE > 1.4), but LightGBM consistently outperforms. The challenge is fundamental: when an item sells 0–3 times per week randomly, even sophisticated models have limited signal.

### Business Implication
For a production system, consider **tiered model deployment**:

- **High-volume, low-intermittency items**: LightGBM with tight prediction intervals → aggressive inventory optimization
- **Sparse/intermittent items**: conservative policy + wider safety stock → prioritize service level over efficiency

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
│   ├── evaluation.py             # WRMSSE proxy, MAE, RMSE metrics
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

1. **Metric is a proxy (not official M5 WRMSSE):**  
   - bottom-level only (item-store)  
   - scale uses full training window (not “active period” only)  
   - weights normalized across bottom series (not per-level across 12 levels)

2. **No hierarchical reconciliation:** Forecasts are made independently per item-store. In production, forecasts should be coherent across the hierarchy (item → department → store → total). Methods like bottom-up aggregation or optimal combination (e.g., MinTrace) ensure consistency.

3. **Point forecasts only:** We predict expected sales but don't quantify uncertainty. For inventory optimization, prediction intervals (e.g., 80% coverage) are essential to set safety stock levels.

4. **Recursive rollout:** One-step recursive forecasting can drift over a 28-day horizon. A high-impact extension is **direct multi-horizon** prediction (28 targets) or a single model with a horizon feature + target shifting.

5. **No external data:** We use only sales, calendar, and price. Real systems incorporate weather, promotions calendar, competitor actions, and macroeconomic indicators.

### Recommended Extensions (high-signal, hiring-relevant)

1. **Weight-decile slice report:** show where you win/lose on the highest-revenue series.
2. **Lag-1 and MA(28) baselines:** verify “simple persistence” vs model for top-weight items.
3. **Active-period scaling:** update `compute_scale()` to exclude pre-launch zeros and document the difference.
4. **Direct multi-horizon LGBM:** improve top-weight stability and reduce rollout drift.

---

## What I Learned

This project reinforced several principles:

1. **Baselines matter**: It’s easy to “win” against a weak baseline. Strong baselines (lag-1, lag-7, moving averages) calibrate whether the model is adding real value.

2. **Aggregate metrics hide heterogeneity**: Overall WRMSSE proxy of 1.03 sounds good, but high-intermittency items still have WRMSSE of 1.48. A production system needs segment-specific strategies.

3. **Feature engineering > model complexity**: Most improvements came from getting lags/rollings/calendar/price features correct and leakage-free, not from “fancier” model classes.

4. **Evaluation protocol is everything**: Rolling-origin backtesting with multiple cutoffs prevents overfitting to a single time period. Low variance across cutoffs (±0.02) increases confidence.

---

## References

- [M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy)
- [M5 Competitors Guide](https://mofc.unic.ac.cy/m5-competition/)
- [1st Place Solution Summary](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163684)
- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd edition. OTexts.

---

## License

MIT
