import pandas as pd 

def build_iv_proxy(rv, forward_days=21):
    iv = rv.shift(-forward_days)
    return pd.Series(iv, index=rv.index, name="iv")

