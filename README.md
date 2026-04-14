COCA COLA 
Stock Market Analysis & ML Prediction

A complete end-to-end Python pipeline for analysing stock market data with technical indicators and predicting price direction using machine learning. All charts are fully interactive (Plotly).

---

Project Overview

This project takes a stock market dataset with pre-computed technical indicators (RSI, MACD, Bollinger Bands, Moving Averages, and more) and walks through every stage — from raw data inspection to a backtested trading strategy — in 13 clear steps.

The goal is to predict whether tomorrow's closing price will be **higher or lower** than today's, using features derived from technical analysis.

---

Dataset

The dataset contains **25 columns** covering price, volume, momentum, volatility, and time features:

| Category | Columns |
|----------|---------|
| Price (OHLC) | `Open`, `High`, `Low`, `Close` |
| Volume | `Volume`, `Volume_MA_20` |
| Trend | `MA_20`, `MA_50`, `MA_200`, `EMA_12`, `EMA_26` |
| Momentum | `MACD`, `MACD_Signal`, `RSI_14` |
| Volatility | `BB_Upper`, `BB_Lower`, `BB_Width` |
| Returns | `Daily_Return_Pct`, `Daily_Range`, `Cumulative_Return_Pct` |
| Time | `Date`, `Year`, `Month`, `Quarter`, `Day_of_Week` |

---

Installation

```bash
pip install pandas numpy scikit-learn xgboost plotly scipy
```

---

Pipeline — Step by Step

### Step 1 — Load & Inspect
Load the CSV, parse dates, sort chronologically, and check data types and missing values.

### Step 2 — Clean the Data
- Drop early rows where `MA_200`, `RSI_14`, and `BB_Upper` are NaN (normal for moving average warm-up periods)
- Convert `Day_of_Week` from string (`'Monday'`) to integer (`0`) if needed

### Step 3 — Exploratory Data Analysis 
Four interactive charts:
- **Price overview** — Candlestick chart with all moving averages, Bollinger Bands, Volume, RSI, and MACD in a single 4-panel figure. Drag to zoom, hover for exact values.
- **Return distribution** — Histogram of daily returns + Q-Q plot to detect fat tails
- **Seasonality** — Average return by month and day of week

### Step 4 — Correlation Analysis *(Interactive Heatmap)*
Interactive heatmap of all numeric features. Hover any cell for the exact correlation value, zoom in to read closely.

> **Key insight:** `MA_20`, `MA_50`, `MA_200`, `EMA_12`, `EMA_26` are all highly correlated with `Close`. Using ratios instead of raw values (done in Step 5) reduces multicollinearity significantly.

### Step 5 — Feature Engineering
30+ engineered features created from raw indicators:

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `close_to_ma20` | `Close / MA_20 - 1` | % deviation from 20-day average |
| `close_to_ma200` | `Close / MA_200 - 1` | Long-term trend position |
| `ma20_to_ma50` | `MA_20 / MA_50 - 1` | Golden/Death cross signal |
| `macd_hist` | `MACD - MACD_Signal` | Momentum strength |
| `pct_b` | `(Close - BB_Lower) / (BB_Upper - BB_Lower)` | Position within Bollinger Bands |
| `volume_ratio` | `Volume / Volume_MA_20` | Unusual volume activity |
| `rsi_overbought` | `RSI > 70` | Overbought zone flag |
| `above_ma200` | `Close > MA_200` | Long-term trend regime |
| `*_lag1/2/3` | Shifted by 1–3 days | Model memory of recent values |
| `return_rolling_mean5` | 5-day rolling return mean | Short-term momentum |

### Step 6 — Target Variable
```
target = 1  if tomorrow's Close > today's Close
target = 0  otherwise
```

### Step 7 — Feature Selection
27 final features covering trend ratios, momentum, volatility, volume, regime flags, lagged values, rolling stats, and calendar features.

### Step 8 — Train / Test Split
**Chronological 80/20 split — never random.**

> Randomly splitting time-series data causes data leakage (the model "sees the future" during training). Always split by date.

### Step 9 — Train 3 Models

| Model | Purpose |
|-------|---------|
| Logistic Regression | Baseline — if complex models don't beat this, features need work |
| Random Forest | Robust to correlated features, provides feature importance |
| XGBoost | Gradient boosting — typically the strongest performer on tabular financial data |

### Step 10 — Evaluate 
- **Confusion matrices** — side by side for all three models, hover cells for exact counts
- **ROC curves** — compare AUC scores across models

### Step 11 — Feature Importance 
Horizontal bar charts for Random Forest and XGBoost. Hover any bar for the exact importance score. Typically, lag features, rolling returns, and ratio-based features rank highest.

### Step 12 — Backtest *(Interactive Plotly)*
Simulates a simple trading strategy: go long when the model predicts an up day, stay out otherwise. Compares cumulative return against a buy-and-hold benchmark.

Reports:
- Total return
- Annualised return
- Sharpe ratio
- Win rate

> **Important:** This backtest ignores transaction costs, slippage, and market impact. Real-world returns will be lower.

### Step 13 — Walk-Forward Cross-Validation
5-fold `TimeSeriesSplit` cross-validation — the proper way to cross-validate time-series models. Reports AUC and accuracy per fold.

---


## 📌 Notes

- Accuracy above **53–55% consistently** is meaningful in financial ML — the market is hard to predict
- A high Sharpe ratio (> 1.0) in backtesting is a good target before considering live trading  
- Always validate on out-of-sample data and account for real-world trading costs before deploying any strategy
