import yfinance as yf
from forecasting.ml_features import build_ml_dataset
from forecasting.walk_forward import walk_forward_xgb
import matplotlib.pyplot as plt 

prices = yf.download("SPY", start="2020-01-01", auto_adjust = True)["Close"]
prices = prices.squeeze()
df = build_ml_dataset(prices)

wf_forecast = walk_forward_xgb(df)
print(wf_forecast.head())
print(wf_forecast.tail())

plt.plot(df["future_vol"], label = "Actual")
plt.plot(wf_forecast, label = "Walk Forward XGB")
plt.legend()
plt.grid()
plt.show()


