import os
import yfinance as yf
import pandas as pd

def get_spot_price(ticker):
    """
    Retrieve latest spot price for a ticker.
    """
    stock = yf.Ticker(ticker)
    history = stock.history(period="1d")
    if history.empty:
        raise ValueError(
            f"No price data found for {ticker}"
        )

    return history["Close"].iloc[-1]

 

def get_expiries(ticker):
    """
    Return available option expiration dates.
    """
    stock = yf.Ticker(ticker)

    return stock.options

def get_option_chain(ticker, expiry):
    """
    Retrieve call and put option chains for a given expiry.
    """
    stock = yf.Ticker(ticker)
    option_chain = stock.option_chain(expiry)
    calls = option_chain.calls
    puts = option_chain.puts
    return calls, puts

def save_option_chain(ticker, expiry):
    """
    Save option chain data to CSV files.
    """
    calls, puts = get_option_chain(
        ticker,
        expiry
    )
    os.makedirs(
        "data/raw",
        exist_ok=True
    )
    calls.to_csv(
        f"data/raw/{ticker}_{expiry}_calls.csv",
        index=False
    )
    puts.to_csv(
        f"data/raw/{ticker}_{expiry}_puts.csv",
        index=False
    )
    print(f"Saved {ticker} {expiry}")