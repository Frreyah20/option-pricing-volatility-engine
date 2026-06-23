import pandas as pd
from xgboost import XGBRegressor

def walk_forward_xgb(df, start_train_size = 756, test_size = 252):
    predictions = []
    prediction_dates = []
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
    y = df["future_vol"]
    start = start_train_size
    while start < len(df):
        end = min(start+test_size, len(df))
        # Purge 20 observations from the end of the training set to prevent overlapping outcomes leakage
        # The target future_vol evaluates 21 days ahead, meaning the last 20 days of the training set 
        # overlap with the strictly unknown future of the test set.
        X_train = X.iloc[:start - 20]
        y_train = y.iloc[:start - 20]
        X_test = X.iloc[start:end]
        print(
            f"Train End={X_train.index[-1]}, "
            f"Test Start={X_test.index[0]}"
        )
        model = XGBRegressor(n_estimators = 200, max_depth = 4, learning_rate = 0.05, random_state = 42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions.extend(preds)
        prediction_dates.extend(X_test.index)
        start +=test_size 

    return pd.Series(predictions, index=prediction_dates, name="walk_forward_forecast")