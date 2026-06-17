def analyze_skew(smile_df, S):
    if smile_df.empty:
        raise ValueError("No valid smile data found")

    otm_put_target = 0.90 * S
    atm_target = S
    otm_call_target = 1.10 * S

    otm_put_idx = (smile_df["strike"] - otm_put_target).abs().idxmin()
    atm_idx = (smile_df["strike"] - atm_target).abs().idxmin()
    otm_call_idx = (smile_df["strike"] - otm_call_target).abs().idxmin()
    put_strike = smile_df.loc[otm_put_idx, "strike"]
    atm_strike = smile_df.loc[atm_idx, "strike"]
    call_strike = smile_df.loc[otm_call_idx, "strike"]

    put_iv = smile_df.loc[otm_put_idx, "implied_vol"]
    atm_iv = smile_df.loc[atm_idx, "implied_vol"]
    call_iv = smile_df.loc[otm_call_idx, "implied_vol"]

    skew = put_iv - call_iv

    return {
        "put_strike": put_strike,
        "atm_strike": atm_strike,
        "call_strike": call_strike,
        "put_iv": put_iv,
        "atm_iv": atm_iv,
        "call_iv": call_iv,
        "skew": skew
    }
    

    