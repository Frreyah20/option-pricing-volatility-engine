import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from forecasting.ml_features import build_ml_dataset, train_random_forest, train_xgboost

def main():
    print("Fetching data and building dataset...")
    prices = yf.download("SPY", start="2020-01-01", auto_adjust=True)["Close"]
    prices = prices.squeeze()
    
    df = build_ml_dataset(prices)
    
    # Replicate the 20-day purged train/test split to extract the out-of-sample test features and targets
    X = df[["ret_1d", "ret_5d", "ret_21d", "hist_vol", "ewma_vol", "garch_vol"]]
    y = df["future_vol"]
    
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    print("Training ML Models...")
    # Get ML forecasts
    _, rf_preds, _ = train_random_forest(df)
    _, xgb_preds, _ = train_xgboost(df)
    
    print("Evaluating Baseline Forecasts...")
    # Extract baseline forecasts directly from out-of-sample test features
    hv_preds = X_test["hist_vol"]
    ewma_preds = X_test["ewma_vol"]
    garch_preds = X_test["garch_vol"]
    
    models = {
        "Historical Volatility": hv_preds,
        "EWMA": ewma_preds,
        "GARCH": garch_preds,
        "Random Forest": rf_preds,
        "XGBoost": xgb_preds
    }
    
    results = []
    
    for name, preds in models.items():
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })
        
    results_df = pd.DataFrame(results)
    # Sort by RMSE (lower is better)
    results_df = results_df.sort_values(by="RMSE").reset_index(drop=True)
    
    print("\n===== FORECASTING BENCHMARK COMPARISON =====")
    print(results_df.to_string(index=False))
    print("============================================")
    
    # Identify best model
    best_model = results_df.iloc[0]["Model"]
    print(f"\nBest Model: {best_model} (Lowest RMSE)")
    
    # Identify the strongest classical baseline
    classical_models = ["Historical Volatility", "EWMA", "GARCH"]
    classical_results = results_df[results_df["Model"].isin(classical_models)]
    strongest_baseline_rmse = classical_results.iloc[0]["RMSE"]
    strongest_baseline_name = classical_results.iloc[0]["Model"]
    
    xgboost_rmse = results_df[results_df["Model"] == "XGBoost"]["RMSE"].values[0]
    
    print(f"Strongest Classical Baseline: {strongest_baseline_name} (RMSE: {strongest_baseline_rmse:.4f})")
    
    # Calculate percentage improvement
    if xgboost_rmse < strongest_baseline_rmse:
        improvement = ((strongest_baseline_rmse - xgboost_rmse) / strongest_baseline_rmse) * 100
        print(f"XGBoost Improvement over {strongest_baseline_name}: {improvement:.2f}%")
    else:
        underperformance = ((xgboost_rmse - strongest_baseline_rmse) / strongest_baseline_rmse) * 100
        print(f"XGBoost UNDERPERFORMS {strongest_baseline_name} by: {underperformance:.2f}%")

if __name__ == "__main__":
    main()
