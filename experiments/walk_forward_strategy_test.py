import yfinance as yf

from forecasting.ml_features import build_ml_dataset
from forecasting.walk_forward import walk_forward_xgb

from strategies.forecast_vol_strategy import (
    compute_forecast_spread,
    generate_forecast_signal,
    plot_forecast_spread
)

from backtests.vol_backtester import (
    run_backtest,
    plot_equity_curve
)

from backtests.performance import (
    sharpe_ratio,
    max_drawdown,
    win_rate,
    annual_return
)

prices = yf.download(
    "SPY",
    start="2020-01-01",
    auto_adjust=True
)["Close"]

prices = prices.squeeze()

df = build_ml_dataset(prices)

wf_forecast = walk_forward_xgb(df)

iv_proxy = df["future_vol"]

common_index = (
    iv_proxy.index
    .intersection(wf_forecast.index)
)

iv_aligned = iv_proxy.loc[common_index]

forecast_aligned = wf_forecast.loc[common_index]

spread_df = compute_forecast_spread(
    iv_aligned,
    forecast_aligned
)

print(spread_df.head())
print(spread_df.tail())

spread_df = generate_forecast_signal(
    spread_df,
    threshold=0.01
)

print(
    spread_df[
        ["spread", "signal"]
    ].tail()
)

returns = prices.pct_change().dropna()

backtest_df = run_backtest(
    spread_df,
    returns
)

plot_equity_curve(
    backtest_df
)

print()
print("===== WALK FORWARD XGB =====")

print(
    "Sharpe =",
    round(
        sharpe_ratio(
            backtest_df["strategy_return"]
        ),
        3
    )
)

print(
    "Max DD =",
    round(
        max_drawdown(
            backtest_df["equity_curve"]
        ),
        3
    )
)

print(
    "Win Rate =",
    round(
        win_rate(
            backtest_df["strategy_return"]
        ),
        3
    )
)

print(
    "Annual Return =",
    round(
        annual_return(
            backtest_df["equity_curve"]
        ),
        3
    )
)