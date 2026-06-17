import numpy as np


def sharpe_ratio(strategy_returns):
    if strategy_returns.std() == 0:
        return 0
    return (strategy_returns.mean()/ strategy_returns.std())* np.sqrt(252)

def max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max)/ running_max
    return drawdown.min()

def win_rate(strategy_returns):
    trades = strategy_returns[strategy_returns != 0]
    if len(trades) == 0:
        return 0
    wins = (trades > 0).sum()
    return wins / len(trades)

def annual_return(equity_curve):
    total_return = (equity_curve.iloc[-1]/ equity_curve.iloc[0])
    years = len(equity_curve) / 252
    return (total_return ** (1 / years)) - 1