import pandas as pd


def apply_regime_filter(signal_df, regime, allowed_regime):
    df = signal_df.copy()
    aligned_regime = regime.reindex(df.index)
    df["signal"] = df["signal"].where(aligned_regime.isin(allowed_regime), 0)
    return df

