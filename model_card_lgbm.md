# Model Card: LightGBM Demand Forecaster

## Model Overview

| Property | Value |
|----------|-------|
| Model Type | Gradient Boosted Decision Trees (LightGBM) |
| Task | 28-day ahead demand forecasting |
| Granularity | Item-store-day |
| Training Data | M5 Walmart sales (2011-2016) |
| Primary Metric | WRMSSE = 1.03 ± 0.02 |

---

## Intended Use

### Primary Use Case
Generate 28-day point forecasts for daily unit sales at the item-store level to support:
- Inventory replenishment planning
- Safety stock optimization
- Demand sensing for supply chain

### Users
- Demand planners
- Inventory managers
- Supply chain analytics teams

### Out of Scope
- Real-time pricing decisions (model doesn't capture price elasticity causally)
- New product forecasting (requires historical data)
- Promotion optimization (treats price as exogenous)

---

## Training Data

### Source
M5 Forecasting Competition dataset (Walmart)

### Composition
- **Time range**: January 2011 - April 2016 (~1,900 days)
- **Items**: 3,049 products
- **Stores**: 10 stores across CA, TX, WI
- **Total observations**: ~58M item-store-days

### Features Used

| Feature | Description | Leakage Risk |
|---------|-------------|--------------|
| lag_28 | Sales 28 days ago | Safe (≥ horizon) |
| lag_35, 42, 49, 56 | Weekly lag multiples | Safe |
| roll_mean_7 | 7-day rolling mean (shifted 28) | Safe |
| roll_mean_14, 28 | Longer rolling means | Safe |
| roll_std_7, 28 | Rolling standard deviation | Safe |
| wday | Day of week (1-7) | Known in advance |
| month | Month (1-12) | Known in advance |
| is_weekend | Saturday/Sunday flag | Known in advance |
| snap | State-specific SNAP benefit day | Known in advance |
| price | Current sell price | Assumed known |
| price_change | 7-day price change | Assumed known |

### Features NOT Used
- `item_id`, `store_id`, `dept_id` (categorical) — caused train/test distribution issues
- `event_type_1` — too sparse, caused NaN problems
- `year` — caused extrapolation issues at forecast time

---

## Model Architecture

### Algorithm
LightGBM (Light Gradient Boosting Machine)

### Hyperparameters

```python
{
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "num_leaves": 255,
    "learning_rate": 0.03,
    "n_estimators": 200,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}
```

### Why Tweedie?
The Tweedie distribution naturally handles:
- Zero-inflated data (47% of observations are zero sales)
- Right-skewed positive values
- Continuous predictions that can be rounded to counts

Variance power of 1.1 is between Poisson (1.0) and Gamma (2.0), appropriate for overdispersed count data.

---

## Evaluation

### Protocol
Rolling-origin backtesting with 3 cutoffs:
- Cutoff 1857 → Forecast 1858-1885
- Cutoff 1885 → Forecast 1886-1913
- Cutoff 1913 → Forecast 1914-1941

### Metrics

| Cutoff | WRMSSE | MAE | RMSE |
|--------|--------|-----|------|
| 1857 | 1.051 | 1.077 | 2.274 |
| 1885 | 1.017 | 1.059 | 2.160 |
| 1913 | 1.013 | 1.075 | 2.204 |
| **Mean** | **1.027** | **1.070** | **2.213** |
| **Std** | **0.021** | **0.010** | **0.057** |

### Comparison to Baseline

| Model | WRMSSE | Improvement |
|-------|--------|-------------|
| Seasonal Naive (lag-7) | 1.318 | — |
| LightGBM | 1.027 | **-22%** |

### Performance by Segment

**By Volume Tier:**
| Tier | WRMSSE | Interpretation |
|------|--------|----------------|
| High | 0.94 | Excellent — beats naive lag-1 |
| Medium | 1.04 | Good — near baseline |
| Low | 1.34 | Acceptable — limited signal |

**By Intermittency:**
| Tier | WRMSSE | Interpretation |
|------|--------|----------------|
| Low (consistent) | 0.87 | Excellent |
| Medium | 1.11 | Good |
| High (sporadic) | 1.48 | Challenging — fundamental noise |

---

## Limitations

### Known Failure Modes

1. **Highly intermittent items**: WRMSSE = 1.48 for items with >50% zero-sales days. The model predicts low values but can't capture sporadic spikes.

2. **New products**: No historical lags available. Would need cold-start handling (e.g., category averages, similar item matching).

3. **Regime changes**: Model trained on 2011-2016 may not generalize to post-COVID shopping patterns without retraining.

4. **Promotional spikes**: Large promotions can cause sales 10x normal. The model smooths these via rolling averages, potentially underpredicting peaks.

### Ethical Considerations

- Model predictions may influence worker scheduling. Systematic underprediction could lead to understaffing and worker stress.
- Overprediction for perishables leads to food waste.
- No direct demographic or personally identifiable information is used.

---

## Deployment Recommendations

### Monitoring
Track weekly:
- Rolling 28-day WRMSSE (alert if > 1.2)
- Bias: mean(actual - predicted) should be near 0
- Coverage: % of actuals within ±50% of prediction

### Retraining Triggers
- WRMSSE sustained above 1.2 for 2+ weeks
- New product category launch
- Major external event (pandemic, competitor entry)

### Integration
```python
from models.lgbm import lgbm_forecast, LGBMConfig

# Generate forecasts
predictions = lgbm_forecast(
    df=sales_data,
    cutoff_day=current_day,
    horizon=28,
    model_cfg=LGBMConfig()
)

# Output: DataFrame with [item_id, store_id, day_index, y_pred]
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-01 | Initial release |

---

## Contact

For questions about this model, contact the repository maintainer.
