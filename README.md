# Option Pricing & Volatility Forecasting Research Platform

## Overview

This project is a quantitative research platform for option pricing, volatility modeling, volatility forecasting, and systematic strategy evaluation.

The platform combines classical derivatives pricing techniques with statistical and machine learning methods to analyze option markets, forecast volatility, and evaluate volatility-informed trading strategies.

The project evolved from a pricing engine into a broader research workflow covering:

- Option Pricing
- Greeks Analytics
- Numerical Pricing Methods
- Implied Volatility Extraction
- Volatility Smile & Skew Analysis
- Volatility Surface Construction
- Volatility Forecasting
- Forecast Model Benchmarking
- Walk-Forward Validation
- Strategy Backtesting
- Cost Sensitivity Analysis
- Volatility Regime Analysis
- Interactive Streamlit Dashboard

---

## Key Results

### Forecast Benchmark Comparison

A formal out-of-sample benchmark was conducted to compare classical volatility models against machine learning approaches.

| Model | RMSE |
|----------|----------:|
| GARCH | 0.110108 |
| EWMA | 0.119561 |
| Historical Volatility | 0.121288 |
| XGBoost | 0.126105 |
| Random Forest | 0.129947 |

**Observation:** Under the selected dataset, feature set, and evaluation framework, GARCH achieved the lowest forecasting error among the models tested. The machine learning models captured some volatility dynamics but did not outperform the strongest classical baseline on this prediction horizon.

### Cost Sensitivity Analysis

Strategy performance was evaluated under increasing execution slippage assumptions.

| Slippage (bps) | Sharpe Ratio | Annual Return | Max Drawdown |
|----------|----------:|----------:|----------:|
| 0 | 2.248 | 36.13% | -7.88% |
| 5 | 1.957 | 30.82% | -9.48% |
| 10 | 1.668 | 25.72% | -11.20% |
| 20 | 1.096 | 16.08% | -18.01% |

**Observation:** Strategy performance deteriorates as execution costs increase, highlighting the importance of realistic transaction cost assumptions during backtesting.

### Validation Improvements

The forecasting framework uses:

- Expanding-window walk-forward validation
- Purged train/test splits
- 20-day embargo period to prevent overlapping-outcomes leakage

This ensures that forecasting performance is evaluated using strictly out-of-sample observations.

---

## Live Application

The project is deployed as an interactive Streamlit dashboard and can be accessed here:

🔗 **Live Demo:** https://option-pricing-volatility-engine-89zfnt4sfzkwwrur5z7mgo.streamlit.app/

---

## Dashboard Modules

### 1. Option Analytics

- Black-Scholes Option Pricing
- Greeks Calculation (Delta, Gamma, Vega, Theta, Rho)
- Interactive parameter controls
- Real-time pricing updates

### 2. Volatility Forecasting

- Historical Volatility
- EWMA Volatility Forecasting
- GARCH(1,1) Volatility Modeling
- Comparative volatility analysis

### 3. ML Volatility Forecasting

- Random Forest Volatility Forecasting
- XGBoost Volatility Forecasting
- Feature Importance Analysis
- Model Performance Comparison

### 4. Trading Strategies

- Volatility Risk Premium Strategy
- Forecast Spread Strategy
- Backtesting Framework
- Performance Metrics (Sharpe Ratio, Drawdown, Win Rate, Annual Return)

### 5. Regime Analysis

- Volatility Regime Classification
- Regime Distribution Analysis
- Regime-Based Strategy Filtering
- Transition Analysis

---

## Dashboard Preview

### Home Page

![Home](screenshots/1.png)

### Option Analytics

![Option Analytics](screenshots/2.png)

### Volatility Forecasting

![Volatility Forecasting](screenshots/3.png)

### Machine Learning Volatility Forecasting

![ML Forecasting](screenshots/4.png)

### Trading Strategies

![Trading Strategies](screenshots/5.png)

### Regime Analysis

![Regime Analysis](screenshots/6.png)

---

## Project Architecture

```text
option_pricing_volatility_engine/

├── pricing/
├── volatility/
├── forecasting/
├── strategies/
├── backtests/
├── research/
├── surface/
├── data/
├── pages/
├── experiments/

├── app.py
├── requirements.txt
└── README.md
```

---

## Features

### Option Pricing Engine

Implemented:

- Black-Scholes Option Pricing
- European Option Pricing
- Put-Call Parity Validation

### Greeks & Risk Analytics

Implemented:

- Delta
- Gamma
- Vega
- Theta
- Rho

### Numerical Pricing Methods

Implemented:

- Binomial Tree Pricing
- Monte Carlo Simulation

Validation included convergence analysis against Black-Scholes benchmarks.

### Implied Volatility Engine

Implemented:

- Bisection Solver
- Newton-Raphson Solver

Market option prices can be converted into implied volatility estimates.

### Volatility Analytics

Implemented:

- Volatility Smile Analysis
- Volatility Skew Analysis
- Term Structure Analysis
- Volatility Surface Construction

### Volatility Forecasting

Implemented:

- Historical Volatility
- EWMA Volatility
- GARCH(1,1)

### Machine Learning Forecasting

Feature Set:

- 1-Day Returns
- 5-Day Returns
- 21-Day Returns
- Historical Volatility
- EWMA Volatility
- GARCH Volatility

Models:

- Random Forest Regressor
- XGBoost Regressor

### Trading Strategies

Implemented:

#### Volatility Risk Premium Strategy

Signal generated from:

```text
Implied Volatility - Realized Volatility
```

#### Forecast Spread Strategy

Signal generated from:

```text
Implied Volatility - Forecast Volatility
```

### Regime Analysis

Volatility environments classified into:

- LOW Volatility
- MEDIUM Volatility
- HIGH Volatility

Research included:

- Regime Performance Analysis
- Regime Filtering
- Regime Transition Analysis

---

## Validation Methodology

### Walk-Forward Evaluation

Forecasting models are evaluated using an expanding-window walk-forward framework that simulates historical deployment.

### Purged Validation

A 20-day purge is applied between training and testing windows to prevent overlapping-outcomes leakage.

The forecasting target uses a 21-day forward volatility horizon. Without a purge, observations near the train-test boundary can inadvertently incorporate information from the testing period, artificially inflating model performance.

---

## Streamlit Dashboard

The platform includes a multi-page Streamlit dashboard.

### Pages

1. Option Analytics
2. Volatility Forecasting
3. ML Volatility Forecasting
4. Trading Strategies
5. Regime Analysis

---

## Technologies Used

- Python
- Streamlit
- NumPy
- Pandas
- SciPy
- Scikit-Learn
- XGBoost
- ARCH
- yfinance
- Matplotlib

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd option_pricing_volatility_engine
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Dashboard

```bash
streamlit run app.py
```

---

## Run Experiments

Forecast Benchmark:

```bash
python -m experiments.benchmark_test
```

Cost Sensitivity Analysis:

```bash
python -m experiments.cost_sensitivity
```

---

## Limitations

- Uses end-of-day Yahoo Finance data rather than intraday market data.
- Volatility surface construction currently relies on interpolation rather than calibrated models such as SABR or SVI.
- Strategy backtests use underlying asset returns as a proxy for volatility-informed allocation decisions rather than a fully delta-hedged options portfolio.

---

## Future Work

Potential extensions include:

- EGARCH and GJR-GARCH Models
- SABR Volatility Smile Calibration
- Heston Stochastic Volatility Model
- Delta-Hedged Options Backtesting
- Portfolio-Level Volatility Strategies

---

## Author

Frreyah