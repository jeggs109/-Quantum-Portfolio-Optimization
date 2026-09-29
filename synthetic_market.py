"""
synthetic_market.py

Generates synthetic-but-realistic daily stock returns for N assets with
a specified correlation structure, using Cholesky decomposition to impose
correlation on independent Gaussian noise -- the same trick used to
simulate correlated Brownian motions in physics.

Why synthetic data? This container has no internet access, so we can't
pull real prices via yfinance. Synthetic data has an upside for an article:
it's fully reproducible (fixed seed) and you control the ground truth
(you KNOW the true mu/covariance you generated from), which makes it much
easier to sanity-check whether your optimizer is doing something sane.

To swap in real data later: replace `generate_synthetic_returns()` with
a call to yfinance, compute daily % returns, then feed them into
`compute_mu_sigma()` unchanged -- everything downstream is agnostic to
where the returns came from.
"""

import numpy as np


def generate_synthetic_returns(n_assets=5, n_days=500, seed=42):
    """
    Returns a (n_days, n_assets) array of daily returns.

    Each asset gets a random annualized drift (expected return) and
    volatility, and assets are correlated via a random correlation matrix
    (built so it's guaranteed positive semi-definite).
    """
    rng = np.random.default_rng(seed)

    # Random annualized drift and volatility per asset (roughly realistic ranges)
    annual_drift = rng.uniform(0.02, 0.18, size=n_assets)      # 2%-18%/yr
    annual_vol = rng.uniform(0.15, 0.45, size=n_assets)        # 15%-45%/yr

    daily_drift = annual_drift / 252
    daily_vol = annual_vol / np.sqrt(252)

    # Build a random but valid correlation matrix:
    # start from random matrix A, form A A^T, normalize to correlation form
    A = rng.normal(size=(n_assets, n_assets))
    cov_raw = A @ A.T
    d = np.sqrt(np.diag(cov_raw))
    corr = cov_raw / np.outer(d, d)

    # Convert correlation -> covariance using our chosen daily vols
    cov_daily = np.outer(daily_vol, daily_vol) * corr

    # Cholesky decomposition to impose correlation on independent noise
    L = np.linalg.cholesky(cov_daily)

    independent_noise = rng.normal(size=(n_days, n_assets))
    correlated_noise = independent_noise @ L.T

    returns = daily_drift + correlated_noise
    return returns


def compute_mu_sigma(returns, annualize=True):
    """
    Given a (n_days, n_assets) returns array, compute expected return
    vector mu and covariance matrix Sigma.
    """
    mu = returns.mean(axis=0)
    sigma = np.cov(returns.T)

    if annualize:
        mu = mu * 252
        sigma = sigma * 252

    return mu, sigma


if __name__ == "__main__":
    returns = generate_synthetic_returns(n_assets=5, n_days=500)
    mu, sigma = compute_mu_sigma(returns)

    names = [f"Asset_{i}" for i in range(len(mu))]
    print("Annualized expected returns:")
    for n, m in zip(names, mu):
        print(f"  {n}: {m:.3f}")

    print("\nAnnualized covariance matrix:")
    print(np.round(sigma, 4))
