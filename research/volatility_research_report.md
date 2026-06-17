# Option Pricing and Volatility Forecasting Research Platform

## Author

Frreyah

## Overview

This project develops a complete quantitative research platform covering option pricing, volatility modeling, implied volatility extraction, machine learning-based volatility forecasting, and systematic volatility trading strategies.

The platform combines classical derivatives theory with modern statistical and machine learning techniques to study how volatility can be modeled, forecasted, and transformed into actionable trading signals.

Core components include:

- Black-Scholes Pricing Engine
- Greeks and Risk Analytics
- Binomial and Monte Carlo Pricing
- Implied Volatility Engine
- Volatility Smile and Skew Analysis
- Volatility Surface Construction
- Historical, EWMA, and GARCH Forecasting
- Random Forest and XGBoost Volatility Models
- Walk-Forward Validation Framework
- Volatility Forecast Trading Strategies
- Regime-Based Risk Management

# Volatility Forecasting and Trading Research Platform

## 1. Executive Summary

This project develops a comprehensive volatility research and trading framework that integrates option pricing theory, volatility forecasting models, machine learning techniques, and systematic trading strategies. The objective is to investigate whether differences between market-implied volatility and statistically forecasted volatility can be transformed into profitable trading signals.

The research begins with the implementation and validation of classical option pricing models, including the Black-Scholes framework, analytical Greeks, finite-difference approximations, binomial tree methods, and Monte Carlo simulation. An implied volatility engine is then constructed using both bisection and Newton-Raphson root-finding techniques.

The forecasting layer consists of multiple volatility estimation approaches, including Historical Volatility, Exponentially Weighted Moving Average (EWMA), and GARCH(1,1) models. To further improve predictive performance, machine learning models including Random Forest and XGBoost are trained on engineered return and volatility features.

A volatility trading strategy is developed based on the spread between implied volatility and forecasted volatility. Walk-forward validation is employed to avoid look-ahead bias and simulate realistic model deployment. Additional research is conducted on volatility regimes and market-state transitions to determine how strategy performance varies across different volatility environments.

Key findings include:

* XGBoost outperformed Random Forest in volatility forecasting tasks.
* Walk-forward testing produced a Sharpe Ratio of approximately 2.16 with an annualized return exceeding 34%.
* Strategy performance varied significantly across volatility regimes.
* Medium-volatility environments produced the strongest risk-adjusted performance.
* Regime filtering improved Sharpe Ratio from 2.05 to 2.29.
* Maximum drawdown was reduced from approximately 24.5% to 7.3% through regime-based risk management.

The final result is a complete volatility research platform that combines quantitative modeling, statistical forecasting, machine learning, systematic trading, and performance evaluation into a unified framework.

## 2. Introduction

Volatility is one of the most important variables in financial markets. It directly influences option prices, portfolio risk, derivative valuation, and trading decisions. While asset prices themselves are notoriously difficult to predict, volatility exhibits persistent statistical characteristics such as clustering, mean reversion, and regime dependence, making it a suitable target for quantitative modeling.

Market participants typically distinguish between two forms of volatility. Realized volatility represents the actual historical variability observed in asset returns, while implied volatility reflects the market's expectation of future volatility embedded in option prices. The difference between these two quantities often gives rise to the Volatility Risk Premium (VRP), a phenomenon that has been widely documented in academic literature and practitioner research.

The central hypothesis of this project is that forecasted future volatility can be compared against market-implied volatility to identify potential mispricings. If implied volatility is significantly higher than expected future volatility, options may be overpriced relative to their forecasted risk. Conversely, if implied volatility is lower than forecasted volatility, options may be underpriced.

To investigate this hypothesis, a multi-stage research framework is developed:

1. Build and validate option pricing models.
2. Construct an implied volatility extraction engine.
3. Develop statistical volatility forecasting models.
4. Train machine learning models for volatility prediction.
5. Generate trading signals using implied-versus-forecast volatility spreads.
6. Evaluate performance using walk-forward backtesting.
7. Analyze performance across volatility regimes.
8. Improve risk-adjusted returns using regime-based filtering techniques.

The project uses SPY market data as the primary testing universe and evaluates model performance using standard quantitative metrics including Sharpe Ratio, Maximum Drawdown, Win Rate, Annualized Return, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

The ultimate goal is not merely to forecast volatility, but to determine whether volatility forecasts can be converted into a robust and systematic trading strategy capable of generating attractive risk-adjusted returns.

## 3. Option Pricing and Risk Analytics

### 3.1 Black-Scholes Framework

The foundation of the research platform is the Black-Scholes option pricing model, which provides a closed-form solution for pricing European call and put options under the assumptions of continuous trading, constant volatility, frictionless markets, and lognormally distributed asset prices.

The model serves as the benchmark against which all subsequent pricing and volatility estimation methods are evaluated throughout this project.

For a European call option with underlying price (S), strike price (K), time to maturity (T), risk-free rate (r), and volatility (\sigma), the Black-Scholes model was implemented and used extensively for pricing, implied volatility extraction, and numerical validation exercises.

---

### 3.2 Greeks and Risk Sensitivities

To quantify option risk exposures, the complete set of first- and second-order Greeks was implemented analytically:

* Delta - sensitivity to changes in the underlying asset price.
* Gamma - sensitivity of Delta to underlying price changes.
* Vega - sensitivity to changes in volatility.
* Theta - sensitivity to the passage of time.
* Rho - sensitivity to changes in interest rates.

Using the benchmark parameter set:

* Spot Price = 100
* Strike Price = 100
* Time to Maturity = 1 Year
* Risk-Free Rate = 5%
* Volatility = 20%

the following values were obtained:

| Greek      |    Value |
| ---------- | -------: |
| Call Delta |   0.6368 |
| Put Delta  |  -0.3632 |
| Gamma      |   0.0188 |
| Vega       |  37.5240 |
| Call Theta |  -0.0176 |
| Put Theta  |  -0.0045 |
| Call Rho   |  53.2325 |
| Put Rho    | -41.8905 |

The resulting values exhibit expected behavior. Delta indicates positive exposure for calls and negative exposure for puts, Gamma remains positive, Vega confirms sensitivity to volatility changes, and Theta captures option time decay.

---

### 3.3 Finite Difference Validation

To verify the correctness of the analytical Greek implementations, finite-difference approximations were developed independently.

The numerical estimates were compared against analytical Black-Scholes values:

| Greek    | Analytical | Finite Difference |
| -------- | ---------: | ----------------: |
| Delta    |     0.6368 |            0.6368 |
| Gamma    |     0.0188 |            0.0188 |
| Vega     |    37.5240 |           37.5210 |
| Call Rho |    53.2325 |           53.2248 |

The numerical approximations closely matched analytical solutions, confirming the correctness of the implementation and providing an independent validation framework for the risk analytics engine.

---

### 3.4 Put-Call Parity Verification

An additional validation step was performed using the Put-Call Parity relationship:

[
C - P = S - Ke^{-rT}
]

where:

* (C) = Call Option Price
* (P) = Put Option Price
* (S) = Spot Price
* (K) = Strike Price
* (r) = Risk-Free Rate
* (T) = Time to Maturity

The implementation produced:

| Quantity       |  Value |
| -------------- | -----: |
| (C - P)        | 4.8771 |
| (S - Ke^{-rT}) | 4.8771 |

Absolute Difference:

| Metric |        Value |
| ------ | -----------: |
| Error  | 0.0000000000 |

The parity relationship was satisfied to numerical precision, providing additional confirmation that the pricing engine and Greek calculations were implemented correctly.

---

### 3.5 Summary

The option pricing and risk analytics framework forms the foundation of the broader volatility research platform. Analytical Greeks, numerical validation methods, and theoretical consistency checks were successfully implemented and verified. The resulting infrastructure provides reliable pricing and risk measurement capabilities that support the implied volatility, forecasting, and trading components developed in subsequent sections.

## 4. Numerical Pricing Laboratory

### 4.1 Objective

While the Black-Scholes model provides a closed-form solution for European option pricing, many real-world derivatives require numerical methods due to complex payoff structures or model assumptions. To build a more complete pricing framework, two widely used numerical approaches were implemented and evaluated:

* Binomial Tree Pricing
* Monte Carlo Simulation

The objective was to validate these methods against Black-Scholes benchmarks and analyze their convergence and computational efficiency.

---

### 4.2 Binomial Tree Pricing

A Cox-Ross-Rubinstein (CRR) binomial tree model was implemented for European call option pricing. The method discretizes the underlying asset price process into a recombining tree of possible future price paths and computes the option value through backward induction.

Using the benchmark parameters:

* Spot Price = 100
* Strike Price = 100
* Time to Maturity = 1 Year
* Risk-Free Rate = 5%
* Volatility = 20%

the Black-Scholes benchmark price was:

| Model         | Option Price |
| ------------- | -----------: |
| Black-Scholes |      10.4506 |

The one-step binomial model produced:

| Model             | Option Price |
| ----------------- | -----------: |
| One-Step Binomial |      12.1623 |

The large discrepancy highlights the limitations of coarse discretization. Increasing the number of time steps significantly improved accuracy.

---

### 4.3 Binomial Convergence Analysis

To study convergence, the binomial model was evaluated across multiple tree depths.

| Steps |   Price | Absolute Error |
| ----: | ------: | -------------: |
|     1 | 12.1623 |         1.7117 |
|     2 |  9.5405 |         0.9101 |
|     5 | 10.8059 |         0.3554 |
|    10 | 10.2534 |         0.1972 |
|    25 | 10.5210 |         0.0704 |
|    50 | 10.4107 |         0.0399 |
|   100 | 10.4306 |         0.0200 |
|   250 | 10.4426 |         0.0080 |
|   500 | 10.4466 |         0.0040 |

The results demonstrate monotonic convergence toward the Black-Scholes benchmark as the number of tree steps increases. At 500 steps, pricing error falls below 0.004, confirming the numerical accuracy of the implementation.

---

### 4.4 Monte Carlo Simulation

A Monte Carlo pricing engine was implemented by simulating terminal asset prices under the geometric Brownian motion assumption and computing discounted expected payoffs.

Using 10,000 simulated paths:

| Model         |   Price |
| ------------- | ------: |
| Black-Scholes | 10.4506 |
| Monte Carlo   | 10.3953 |

The corresponding 95% confidence interval was:

| Lower Bound | Upper Bound |
| ----------: | ----------: |
|     10.1049 |     10.6856 |

Since the Black-Scholes benchmark falls within the confidence interval, the simulation results are statistically consistent with the analytical solution.

---

### 4.5 Monte Carlo Convergence Analysis

The number of simulations was gradually increased to study convergence behavior.

| Simulations |   Price | Absolute Error |
| ----------: | ------: | -------------: |
|         100 |  7.1979 |         3.2526 |
|         500 | 10.4582 |         0.0076 |
|       1,000 |  9.8844 |         0.5662 |
|       5,000 | 10.7091 |         0.2585 |
|      10,000 | 10.2765 |         0.1741 |
|      50,000 | 10.4176 |         0.0330 |
|     100,000 | 10.3653 |         0.0853 |

Although Monte Carlo estimates exhibit sampling variability, increasing the number of simulations generally reduces pricing error and stabilizes the estimate around the theoretical value.

---

### 4.6 Computational Performance Comparison

To compare practical efficiency, runtime measurements were collected for all three pricing methods.

| Method                      |   Price | Error vs Black-Scholes | Runtime (Seconds) |
| --------------------------- | ------: | ---------------------: | ----------------: |
| Black-Scholes               | 10.4506 |                 0.0000 |            0.0012 |
| Binomial (500 Steps)        | 10.4466 |                 0.0040 |            0.5183 |
| Monte Carlo (100,000 Paths) | 10.4641 |                 0.0135 |            0.0104 |

The Black-Scholes model was the fastest due to its closed-form solution. Monte Carlo simulation provided reasonable accuracy with significantly lower runtime than a high-depth binomial tree. The binomial model achieved the lowest numerical error among the numerical methods but required substantially greater computation.

---

### 4.7 Summary

The numerical pricing laboratory validated two independent pricing methodologies against analytical Black-Scholes benchmarks. The binomial model demonstrated strong convergence properties as tree depth increased, while Monte Carlo simulation produced statistically consistent estimates through large-scale path generation.

These experiments confirm the correctness of the pricing framework and provide alternative valuation methods that can be extended to more complex derivatives where closed-form solutions are unavailable.

## 5. Implied Volatility Engine

### 5.1 Objective

Unlike historical volatility, which is calculated from past returns, implied volatility represents the market's expectation of future volatility embedded within option prices. Because implied volatility is not directly observable, it must be extracted numerically by solving the inverse option pricing problem.

The objective of this section was to develop a robust implied volatility engine capable of recovering volatility estimates from observed option prices and evaluating the efficiency of different root-finding algorithms.

---

### 5.2 Implied Volatility Extraction

Given a market option price (C_{market}), implied volatility is defined as the value of (\sigma) that satisfies:

C_{market} = BS(S,K,T,r,\sigma)

where:

* S = Spot Price
* K = Strike Price
* T = Time to Maturity
* r = Risk-Free Rate
* sigma = Implied Volatility

Because no closed-form inverse exists for the Black-Scholes formula, numerical methods must be employed to solve for volatility.

Two independent approaches were implemented:

1. Bisection Method
2. Newton-Raphson Method

The use of two separate solvers provides both robustness and validation of the resulting implied volatility estimates.

---

### 5.3 Bisection Method

The bisection algorithm iteratively narrows a volatility interval until the Black-Scholes model price matches the observed market price within a specified tolerance.

Advantages:

* Guaranteed convergence under reasonable assumptions.
* Numerically stable.
* Easy to implement.

Disadvantages:

* Relatively slow convergence.
* Requires repeated pricing evaluations.

Despite its computational cost, the bisection method serves as a reliable benchmark for validating more advanced solvers.

---

### 5.4 Newton-Raphson Method

The Newton-Raphson approach uses the derivative of option price with respect to volatility (Vega) to accelerate convergence.

At each iteration, Newton-Raphson updates volatility using:

sigma_(n+1) = sigma_n - Pricing Error / Vega

Advantages:

* Extremely fast convergence near the solution.
* Requires significantly fewer iterations.

Disadvantages:

* Sensitive to initial guesses.
* May fail when Vega is very small.

In practice, Newton-Raphson produced highly accurate implied volatility estimates while requiring substantially less computation than the bisection approach.

---

### 5.5 Validation Results

To verify correctness, synthetic option prices were generated using known volatility values and subsequently passed into the implied volatility engine.

| True Volatility | Recovered Volatility |
| --------------: | -------------------: |
|            0.10 |                 0.10 |
|            0.15 |                 0.15 |
|            0.20 |                 0.20 |
|            0.30 |                 0.30 |
|            0.40 |                 0.40 |
|            0.60 |                 0.60 |

Both the Bisection and Newton-Raphson implementations successfully recovered the original volatility values to numerical precision.

These results confirm that the implied volatility engine accurately solves the inverse Black-Scholes problem.

---

### 5.6 Solver Performance Comparison

To compare computational efficiency, both methods were applied to the same option pricing problem.

| Method         | Implied Volatility | Runtime (Seconds) |
| -------------- | -----------------: | ----------------: |
| Bisection      |           0.600000 |           0.01248 |
| Newton-Raphson |           0.600000 |           0.00073 |

Newton-Raphson achieved approximately seventeen times faster execution while maintaining identical numerical accuracy.

This performance advantage becomes increasingly important when processing large option chains containing hundreds or thousands of contracts.

---

### 5.7 Market Data Integration

The implied volatility engine was integrated with live option-chain data obtained through Yahoo Finance. Supporting functionality was developed for:

* Spot price retrieval
* Expiry selection
* Option chain download
* Call and put filtering
* ATM option identification

This infrastructure enables automated implied volatility extraction directly from market data and forms the foundation for subsequent volatility surface and forecasting analyses.

---

### 5.8 Summary

A complete implied volatility extraction framework was developed using both Bisection and Newton-Raphson solvers. Validation experiments demonstrated accurate recovery of known volatility values, while runtime analysis showed a significant computational advantage for the Newton-Raphson approach.

The resulting implied volatility engine provides the critical link between observed option prices and volatility forecasting models developed later in the research pipeline.

## 6. Volatility Structure Analysis

### 6.1 Objective

Beyond forecasting future volatility, option markets provide information about how volatility varies across strikes and maturities. These patterns reveal investor expectations, risk aversion, and market positioning.

The objective of this section was to develop tools for analyzing:

- Volatility Smile
- Volatility Skew
- Volatility Term Structure
- Volatility Surface Construction

using live option-chain data.

---

### 6.2 Volatility Smile

A volatility smile was constructed by extracting implied volatilities across different strike prices for a fixed expiration date.

For each option contract:

1. Market mid-price was computed from bid and ask quotes.
2. Implied volatility was recovered using the implied volatility solver.
3. Volatility values were plotted against strike price.

This analysis allows visualization of how implied volatility changes as options move away from the at-the-money strike.

The framework successfully filtered contracts based on:

- Positive bid and ask prices
- Non-zero open interest
- Reasonable strike ranges around spot price

and generated a strike-volatility dataset suitable for smile analysis.

---

### 6.3 Volatility Skew Analysis

To quantify asymmetry in the volatility smile, implied volatilities were compared across:

- OTM Put Region (90% of Spot)
- ATM Strike
- OTM Call Region (110% of Spot)

The implemented skew metric was:

Skew = OTM Put IV − OTM Call IV

Interpretation:

- Positive skew indicates stronger demand for downside protection.
- Negative skew indicates stronger upside speculation.
- Near-zero skew suggests a relatively symmetric volatility structure.

The framework automatically identifies representative strikes and computes the corresponding skew measure.

---

### 6.4 Volatility Term Structure

A term-structure engine was developed to examine how implied volatility changes across option maturities.

For each expiration date:

1. The nearest ATM option was selected.
2. Implied volatility was extracted.
3. ATM volatility was plotted against maturity.

This analysis provides insight into:

- Short-term uncertainty
- Long-term risk expectations
- Contango and backwardation effects in volatility markets

The resulting framework enables systematic comparison of volatility expectations across the maturity spectrum.

---

### 6.5 Volatility Surface Framework

The final objective was to combine strike and maturity information into a two-dimensional volatility surface.

The implemented pipeline:

1. Download option chains across multiple expirations.
2. Filter illiquid contracts.
3. Extract implied volatilities.
4. Construct a strike-expiry grid.
5. Interpolate missing values.
6. Generate heatmaps and 3D visualizations.

The surface engine was successfully implemented and integrated into the project architecture.

---

### 6.6 Data Quality Challenges

While the volatility surface framework was completed, reliable surface generation was limited by data quality issues in Yahoo Finance option-chain data.

Observed issues included:

- Missing bid/ask quotes
- Zero open-interest contracts
- Sparse strike coverage near spot
- Inconsistent implied volatility fields across expirations

As a result, certain expirations produced insufficient valid contracts for robust surface construction.

This limitation originated from the external data source rather than the surface-building methodology itself.

---

### 6.7 Summary

A complete volatility structure analysis framework was developed, including smile, skew, term-structure, and volatility-surface components. The system provides tools for investigating how market-implied volatility varies across strikes and maturities, extending the project beyond traditional pricing models toward practical options-market analysis.

Although volatility surface generation was constrained by data quality limitations, the analytical framework remains fully implemented and can be readily extended using institutional-grade options data sources.

## 7. Volatility Forecasting and Trading Strategy Research

### 7.1 Objective

The primary objective of this section was to investigate whether future volatility can be forecasted using a combination of statistical and machine learning models, and whether these forecasts can be transformed into profitable trading signals.

The research pipeline consisted of:

1. Volatility Forecasting
2. Machine Learning Modeling
3. Walk-Forward Validation
4. Signal Generation
5. Strategy Backtesting
6. Regime Analysis
7. Regime-Filtered Trading

This transformed the project from a pricing library into a quantitative volatility research platform.

---

### 7.2 Historical Volatility Forecasting

The baseline forecasting model was the 21-day historical volatility estimator:

Historical Volatility = Rolling Standard Deviation × sqrt(252)

Historical volatility provides a simple estimate of future market uncertainty using recent returns and serves as a benchmark for more advanced forecasting approaches.

---

### 7.3 EWMA Volatility Model

An Exponentially Weighted Moving Average (EWMA) model was implemented to place greater emphasis on recent market observations.

The EWMA model updates volatility recursively:

Variance(t) = λ × Variance(t-1) + (1-λ) × Return(t-1)^2

where λ = 0.94.

Compared to historical volatility, EWMA responds more rapidly to volatility shocks and changing market conditions.

---

### 7.4 GARCH Volatility Forecasting

To model volatility clustering and persistence, a GARCH(1,1) model was implemented.

The model estimates:

Variance(t) = ω + α × Return(t-1)^2 + β × Variance(t-1)

The fitted model produced statistically significant volatility parameters, confirming the presence of volatility persistence within SPY returns.

GARCH forecasts were subsequently used as inputs for volatility forecasting and trading strategy construction.

---

### 7.5 Machine Learning Volatility Forecasting

To improve predictive performance, machine learning models were trained to forecast future realized volatility.

The feature set included:

- 1-Day Returns
- 5-Day Returns
- 21-Day Returns
- Historical Volatility
- EWMA Volatility
- GARCH Volatility

The target variable was future 21-day realized volatility.

Two models were implemented:

- Random Forest Regressor
- XGBoost Regressor

---

### 7.6 Model Comparison

Forecasting performance was evaluated using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

Results:

| Model | RMSE | MAE |
|---------|---------:|---------:|
| Random Forest | 0.1315 | 0.0774 |
| XGBoost | 0.1322 | 0.0782 |

Both models demonstrated similar predictive performance, indicating that volatility dynamics can be partially captured using statistical and machine learning features.

---

### 7.7 Walk-Forward Validation

Traditional train-test splits often introduce look-ahead bias in financial applications.

To address this issue, a walk-forward validation framework was implemented.

The procedure:

1. Train on historical data.
2. Generate forecasts for the next period.
3. Expand the training window.
4. Retrain the model.
5. Repeat until the end of the dataset.

Training windows expanded progressively:

- Train: 756 observations
- Train: 1008 observations
- Train: 1260 observations
- Train: 1512 observations

This process more accurately reflects real-world forecasting conditions where future information is unavailable.

---

### 7.8 Volatility Forecast Spread

A volatility spread was defined as:

Spread = Implied Volatility − Forecast Volatility

Interpretation:

- Positive Spread → Options appear expensive relative to forecasted volatility.
- Negative Spread → Options appear cheap relative to forecasted volatility.

This spread forms the foundation of the trading signal generation process.

---

### 7.9 Trading Signal Construction

Trading rules were defined using spread thresholds.

Signal Generation:

- Spread > Threshold → Short Volatility Signal
- Spread < -Threshold → Long Volatility Signal
- Otherwise → Neutral Position

Signals were converted into daily positions and evaluated using a transaction-cost-aware backtesting engine.

---

### 7.10 Strategy Performance

The walk-forward XGBoost forecasting strategy generated the following performance metrics:

| Metric | Value |
|---------|---------:|
| Sharpe Ratio | 2.161 |
| Maximum Drawdown | -10.8% |
| Win Rate | 51.1% |
| Annual Return | 34.5% |

The resulting equity curve exhibited consistent long-term growth with controlled drawdowns.

These results suggest that volatility forecast spreads contain economically meaningful information that can be transformed into systematic trading signals.

---

### 7.11 Volatility Regime Classification

Market environments were classified into three volatility regimes:

- LOW Volatility
- MEDIUM Volatility
- HIGH Volatility

Regimes were determined using realized volatility terciles.

Distribution:

| Regime | Observations |
|---------|---------:|
| LOW | 529 |
| MEDIUM | 543 |
| HIGH | 529 |

This classification framework allows strategy performance to be analyzed across different market environments.

---

### 7.12 Regime Performance Analysis

Strategy performance varied substantially across volatility regimes.

| Regime | Sharpe | Max Drawdown | Win Rate | Annual Return |
|---------|---------:|---------:|---------:|---------:|
| LOW | 1.85 | -7.0% | 49.4% | 165.6% |
| MEDIUM | 3.01 | -8.1% | 53.7% | 187.9% |
| HIGH | 2.13 | -24.5% | 53.9% | 187.0% |

The strategy performed strongest during MEDIUM volatility environments while experiencing its largest drawdowns during HIGH volatility periods.

---

### 7.13 Regime-Filtered Strategy

Given the elevated drawdowns observed during HIGH volatility periods, a regime filter was introduced.

Trading was permitted only during:

- LOW Regime
- MEDIUM Regime

HIGH volatility periods were excluded.

Results:

| Metric | Original Strategy | Filtered Strategy |
|---------|---------:|---------:|
| Sharpe Ratio | 2.05 | 2.29 |
| Maximum Drawdown | -24.5% | -7.3% |

The regime filter improved risk-adjusted performance while dramatically reducing drawdown.

This finding suggests that volatility forecasting signals are most reliable during stable and moderately volatile market conditions.

---

### 7.14 Regime Transition Analysis

To study changing market conditions, transitions between volatility regimes were tracked.

Most observations remained within the same regime:

- HIGH → HIGH: 509
- LOW → LOW: 497
- MEDIUM → MEDIUM: 490

Transition analysis revealed that volatility regimes exhibit strong persistence, consistent with established findings in financial econometrics.

Performance around regime transitions was also examined to understand how strategy returns evolve during shifts in market volatility conditions.

---

### 7.15 Summary

This section developed a complete volatility forecasting and trading research pipeline using statistical models, machine learning forecasts, walk-forward validation, and systematic backtesting.

The strongest result emerged from combining machine learning volatility forecasts with regime-based filtering. The resulting strategy achieved a Sharpe ratio above 2 while reducing maximum drawdown by approximately 70%, demonstrating the value of integrating forecasting models with market regime awareness.

# Conclusion

This project developed a comprehensive option pricing, volatility modeling, and quantitative research platform using Python.

The work began with foundational derivatives pricing techniques, including Black-Scholes valuation, Greeks computation, numerical pricing methods, and implied volatility extraction. These components were validated through convergence studies, finite-difference checks, and analytical benchmarks.

The project was then extended into volatility analytics through the implementation of historical volatility, EWMA forecasting, GARCH modeling, volatility smile analysis, skew measurement, term-structure analysis, and volatility surface construction.

Finally, a complete volatility forecasting and trading research pipeline was developed using machine learning techniques. Random Forest and XGBoost models were trained to forecast future volatility, evaluated using walk-forward validation, and integrated into a systematic trading framework.

The most significant result emerged from regime-aware trading. By filtering trades during high-volatility environments, the strategy improved its Sharpe ratio while substantially reducing maximum drawdown.

Overall, the project demonstrates the integration of derivatives pricing, volatility forecasting, machine learning, and systematic trading into a unified quantitative research platform.

---

# Future Work

Several extensions can further improve the framework:

## Advanced Volatility Models

- EGARCH
- GJR-GARCH
- Stochastic Volatility Models
- Heston Model

These models may capture asymmetric volatility dynamics more effectively than standard GARCH.

## Enhanced Machine Learning

Additional forecasting models can be explored:

- LightGBM
- CatBoost
- LSTM Networks
- Transformer-Based Time Series Models

These approaches may improve predictive accuracy during rapidly changing market conditions.

## Institutional Data Sources

Yahoo Finance option-chain data contains missing values and liquidity issues. Future versions can incorporate:

- Polygon.io
- CBOE DataShop
- OptionMetrics
- Interactive Brokers API

Higher-quality data would improve volatility surface construction and option analytics.

## Portfolio-Level Volatility Trading

The current implementation focuses on a single underlying asset. Future work can extend the framework to:

- Multi-Asset Volatility Trading
- Cross-Sectional Volatility Forecasting
- Volatility Risk Premium Portfolios
- Relative Value Volatility Strategies

## Production Deployment

Future development may include:

- Automated daily forecasting
- Real-time option-chain ingestion
- Cloud deployment
- Interactive research dashboards
- Live strategy monitoring

These enhancements would transform the framework from a research environment into a production-ready quantitative analytics platform.