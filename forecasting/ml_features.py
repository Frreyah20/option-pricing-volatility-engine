import numpy as np 
from forecasting.ewma import ewma_volatility
import pandas as pd  
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from forecasting.garch import garch_volatility, fit_garch
from xgboost import XGBRegressor




def build_ml_dataset(prices):
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    df = pd.DataFrame()
    returns = np.log(prices/prices.shift(1))
    df["ret_1d"] = returns
    df["ret_5d"] = prices.pct_change(5, fill_method=None)
    df["ret_21d"] = prices.pct_change(21, fill_method=None)
    df["hist_vol"] = returns.rolling(21).std() * np.sqrt(252)
    df["ewma_vol"] = ewma_volatility(returns)
    results = fit_garch(returns)
    garch_vol = garch_volatility(results)
    df["garch_vol"] = garch_vol.reindex(df.index) 
    future_vol = returns.rolling(21).std().shift(-21) * np.sqrt(252)
    df["future_vol"] = future_vol
    df = df.dropna()
    return df 

def train_random_forest(df):
    X = df[["ret_1d", "ret_5d", "ret_21d", "hist_vol", "ewma_vol", "garch_vol"]]
    y = df["future_vol"]
    split_idx = int(len(X) * 0.8)
    # Purge 20 observations from the end of the training set to prevent overlapping outcomes leakage
    # This prevents the model from peeking into the strictly unknown test set future.
    X_train, X_test = X.iloc[:split_idx - 20], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx - 20], y.iloc[split_idx:]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    #print("Random Forest RMSE = ", rmse)
    mae = mean_absolute_error(y_test, preds)
    #print("Random Forest MAE = ", mae)
    return model, preds, y_test

def plot_predictions(rf_preds, y_test, xgb_preds):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual")
    plt.plot(rf_preds, label="Random Forest")
    plt.plot(xgb_preds, label="XGBoost")
    plt.title("Volatility Forecast Comparison") 
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid()
    plt.show()  

def plot_feature_importance(model, feature_names):

    importance = pd.Series(model.feature_importances_,index=feature_names).sort_values()
    importance.plot(kind="barh")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.show()  

def train_xgboost(df):
    X = df[["ret_1d", "ret_5d", "ret_21d", "hist_vol", "ewma_vol", "garch_vol"]]
    y = df["future_vol"]
    split_idx = int(len(X) * 0.8)
    # Purge 20 observations from the end of the training set to prevent overlapping outcomes leakage
    # This prevents the model from peeking into the strictly unknown test set future.
    X_train, X_test = X.iloc[:split_idx - 20], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx - 20], y.iloc[split_idx:]
    model = XGBRegressor(n_estimators = 200,max_depth = 4, learning_rate = 0.05, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    #print("XGBoost RMSE = ", rmse)
    mae = mean_absolute_error(y_test, preds)
    #print("XGBoost MAE = ", mae)
    return model, preds, y_test

def plot_xgb_importance(model, feature_names):
    importance = pd.Series(model.feature_importances_, index = feature_names).sort_values()
    importance.plot(kind="barh")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.show()

def compare_models(y_test, rf_preds, xgb_preds):
    rf_rmse = np.sqrt(mean_squared_error(y_test,rf_preds))
    xgb_rmse = np.sqrt(mean_squared_error(y_test,xgb_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    results = pd.DataFrame({
        "Model": [
            "Random Forest",
            "XGBoost"
        ],
        "RMSE": [
            rf_rmse,
            xgb_rmse
        ],
        "MAE": [
            rf_mae,
            xgb_mae
        ]
    })

    print(results)
    return results

def get_rf_forecast_series(model, df):
    X = df[
        [
            "ret_1d",
            "ret_5d",
            "ret_21d",
            "hist_vol",
            "ewma_vol",
            "garch_vol"
        ]
    ]
    forecasts = model.predict(X)
    return pd.Series(forecasts, index=df.index)

def get_xgb_forecast_series(model, df):

    X = df[
        [
            "ret_1d",
            "ret_5d",
            "ret_21d",
            "hist_vol",
            "ewma_vol",
            "garch_vol"
        ]
    ]
    forecasts = model.predict(X)
    return pd.Series(forecasts, index=df.index)
    