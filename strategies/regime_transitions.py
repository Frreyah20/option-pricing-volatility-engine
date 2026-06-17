import pandas as pd


def identify_transitions(regime):
    transitions = pd.DataFrame(index=regime.index)
    transitions["regime"] = regime
    transitions["prev_regime"] = regime.shift(1)
    transitions["transition"] = (
        transitions["prev_regime"].astype(str)
        + " -> "
        + transitions["regime"].astype(str)
    )
    return transitions

def transition_performance(transitions, returns, horizon=5):

    results = []
    unique_transitions = (transitions["transition"].dropna().unique())
    for t in unique_transitions:
        dates = transitions[transitions["transition"] == t].index
        future_returns = []
        for date in dates:
            if date not in returns.index:
                continue
            idx = returns.index.get_loc(date)
            if idx + horizon >= len(returns):
                continue
            r = (returns.iloc[idx+1:idx+horizon+1].sum())
            future_returns.append(r)
        if len(future_returns) == 0:
            continue
        results.append({
            "Transition": t,
            "Count": len(future_returns),
            "AvgReturn":
                pd.Series(
                    future_returns
                ).mean()
        })

    return (pd.DataFrame(results).sort_values("AvgReturn",ascending=False))