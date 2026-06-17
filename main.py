def main():
    TICKERS = [
        "SPY",
        "QQQ",
        "AAPL",
        "NVDA"
    ]
    
'''
for ticker in TICKERS:

    expiries = get_expiries(
        ticker
    )

    first_expiry = expiries[0]

    save_option_chain(
        ticker,
        first_expiry
    )
'''


if __name__ == "__main__":
    main()
