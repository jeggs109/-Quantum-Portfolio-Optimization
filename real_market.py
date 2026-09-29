"""
real_market.py

Drop-in replacement for synthetic_market.py's data source, using real
historical prices via yfinance. Everything downstream (qubo_portfolio.py,
qaoa_scratch.py, main.py) is unchanged -- they only ever consume the
`returns` array and the `mu, sigma` computed from it.

Requires: pip install yfinance
(This container has no internet access, so this file can't be run here --
run it on your own machine.)
"""

import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None


def get_real_returns(tickers, start="2022-01-01", end="2024-01-01"):
    """
    Downloads daily close prices for the given tickers and returns:
        returns : (n_days, n_assets) array of daily % returns
        names   : list of ticker names, in the same column order as `returns`

    This mirrors the shape/format of synthetic_market.generate_synthetic_returns(),
    so compute_mu_sigma() and everything downstream works unchanged.
    """
    if yf is None:
        raise ImportError(
            "yfinance is not installed. Run: pip install yfinance"
        )

    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]

    # yf.download returns a Series instead of a DataFrame if you pass a
    # single ticker -- normalize to DataFrame so .columns always works.
    if data.ndim == 1:
        data = data.to_frame(name=tickers[0])

    # Drop any tickers that came back empty (bad symbol, delisted, etc.)
    missing = [t for t in tickers if t not in data.columns]
    if missing:
        print(f"Warning: no data returned for {missing}, dropping them.")

    data = data.dropna(axis=1, how="all")
    returns = data.pct_change().dropna()

    if returns.isna().any().any():
        # Leftover gaps (e.g. one ticker missing a few days) -- forward-fill
        # prices before computing returns rather than dropping whole rows,
        # so one spotty ticker doesn't shrink your sample for everyone.
        print("Warning: some gaps in the data; forward-filling prices.")
        data = data.ffill().dropna()
        returns = data.pct_change().dropna()

    return returns.values, list(data.columns)


def compute_mu_sigma(returns, annualize=True):
    """
    Identical to synthetic_market.compute_mu_sigma -- repeated here so
    real_market.py can be used standalone without importing the
    synthetic module.
    """
    mu = returns.mean(axis=0)
    sigma = np.cov(returns.T)

    if annualize:
        mu = mu * 252
        sigma = sigma * 252

    return mu, sigma


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "NVDA"]
    returns, names = get_real_returns(tickers, start="2022-01-01", end="2024-01-01")
    mu, sigma = compute_mu_sigma(returns)

    print(f"Tickers used (in order): {names}")
    print(f"Trading days: {returns.shape[0]}")

    print("\nAnnualized expected returns:")
    for n, m in zip(names, mu):
        print(f"  {n}: {m:.3f}")

    print("\nAnnualized covariance matrix:")
    print(np.round(sigma, 4))
