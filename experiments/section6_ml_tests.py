from forecasting.ml_features import build_ml_dataset, train_random_forest, plot_predictions, plot_feature_importance, train_xgboost, plot_xgb_importance, compare_models
import yfinance as yf

prices = yf.download("SPY", start="2020-01-01", auto_adjust=True)["Close"]
df = build_ml_dataset(prices)
rf_model, rf_preds, y_test = train_random_forest(df)

plot_feature_importance(rf_model,["ret_1d", "ret_5d", "ret_21d","hist_vol", "ewma_vol", "garch_vol"])

xgb_model, xgb_preds, _ = train_xgboost(df)
plot_xgb_importance(xgb_model,["ret_1d", "ret_5d", "ret_21d","hist_vol", "ewma_vol", "garch_vol"])
plot_predictions(rf_preds, y_test, xgb_preds)
compare_models(y_test, rf_preds, xgb_preds)