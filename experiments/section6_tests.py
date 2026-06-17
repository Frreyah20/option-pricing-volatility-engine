import yfinance as yf 
from forecasting.historical import log_returns, historical_volatility, plot_historical_volatility
from forecasting.ewma import ewma_volatility, plot_ewma_volatility
from forecasting.garch import fit_garch, garch_volatility, plot_garch_volatility 
from data.market_data import get_expiries, get_option_chain, get_spot_price
from volatility.implied_volatility_solver import implied_vol_bisection
from forecasting.iv_comparison import compare_vols, plot_comparison
from datetime import datetime


prices = yf.download("SPY", start="2020-01-01", auto_adjust=True)["Close"]
returns = log_returns(prices)
returns = returns.squeeze()
hv = historical_volatility(returns, 21)
print(hv.tail())
plot_historical_volatility(hv)

ewma_vol = ewma_volatility(returns)
print(ewma_vol.tail())
plot_ewma_volatility(ewma_vol)

results = fit_garch(returns)
print(results.summary())
garch_vol = garch_volatility(results)
print(garch_vol.tail())
plot_garch_volatility(garch_vol)

hist_vol = hv.iloc[-1]
ewma_vol = ewma_vol.iloc[-1]
garch_vol = garch_vol.iloc[-1]
comparison_df = compare_vols(hist_vol, ewma_vol, garch_vol)
print(comparison_df)
plot_comparison(comparison_df)






