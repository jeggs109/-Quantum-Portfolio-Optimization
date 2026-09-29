"""
main.py

Ties everything together:
1. Generate a synthetic market (5 assets)
2. Solve the CONTINUOUS mean-variance problem classically (textbook finance)
3. Formulate the discretized/cardinality-constrained version as a QUBO
4. Solve the QUBO exactly by brute force (our ground truth)
5. Solve the SAME QUBO with QAOA (our from-scratch quantum simulator)
6. Compare results, and sweep problem size to see how QAOA's success
   probability degrades as n grows (this is the honest, interesting
   finding for the article -- not "quantum wins")

Produces two plots:
  - qaoa_convergence.png : QAOA cost vs optimizer iteration for one run
  - qaoa_vs_n.png        : success probability (finding the true optimum)
                           vs number of assets/qubits
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize

from synthetic_market import generate_synthetic_returns, compute_mu_sigma
from qubo_portfolio import build_qubo, qubo_to_ising, brute_force_solve, qubo_objective
from qaoa_scratch import run_qaoa, most_likely_bitstrings, qaoa_expectation, build_cost_diagonal

# Set to True to use real prices via yfinance instead of synthetic data.
# Requires internet access and `pip install yfinance` -- this container
# has neither, so USE_REAL_DATA must stay False here. On your own machine,
# flip this to True and it swaps in transparently: the QUBO/QAOA/plot code
# below never changes, since it only consumes `mu` and `sigma`.
USE_REAL_DATA = True
REAL_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "NVDA"]


def get_market_data(n_assets, seed=0):
    """
    Single entry point for market data. Returns (mu, sigma, names).
    Swap USE_REAL_DATA above to switch sources -- nothing else in this
    file needs to change.
    """
    if USE_REAL_DATA:
        from real_market import get_real_returns
        if n_assets > len(REAL_TICKERS):
            raise ValueError(
                f"REAL_TICKERS only has {len(REAL_TICKERS)} symbols; "
                f"add more tickers to run with n_assets={n_assets}."
            )
        tickers = REAL_TICKERS[:n_assets]
        returns, names = get_real_returns(tickers, start="2022-01-01", end="2026-01-01")
    else:
        returns = generate_synthetic_returns(n_assets=n_assets, n_days=500, seed=seed)
        names = [f"Asset_{i}" for i in range(n_assets)]

    mu, sigma = compute_mu_sigma(returns)
    return mu, sigma, names


def classical_mean_variance(mu, sigma, risk_aversion=1.0):
    """
    Textbook (continuous, unconstrained-cardinality) mean-variance
    optimization: minimize q * w^T Sigma w - mu^T w subject to sum(w)=1, w>=0.
    This is the "no quantum, no discretization" baseline -- what a
    finance person would actually do in practice.
    """
    n = len(mu)
    x0 = np.ones(n) / n

    def objective(w):
        return risk_aversion * w @ sigma @ w - mu @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n

    res = scipy_minimize(objective, x0, bounds=bounds, constraints=constraints)
    return res.x, objective(res.x)


def run_single_experiment(n_assets, budget, p=2, seed=0, verbose=True, n_restarts=8):
    """
    Runs the full pipeline once for a given number of assets.

    Returns whether QAOA's most likely bitstring matches the true optimum,
    AND the objective-value gap between QAOA's answer and the true optimum
    -- a strict yes/no match is a harsh metric, since a "wrong" bitstring
    can still be a nearly-as-good portfolio (ties/near-ties are common
    once the budget constraint narrows things down).
    """
    mu, sigma, names = get_market_data(n_assets, seed=seed)

    Q, const, penalty = build_qubo(mu, sigma, budget=budget, risk_aversion=1.0)
    h, J, offset = qubo_to_ising(Q, const)

    true_x, true_val = brute_force_solve(Q, const)

    state, params, qaoa_val, cost_diag = run_qaoa(h, J, n_assets, p=p, n_restarts=n_restarts, seed=seed)
    top_bits = most_likely_bitstrings(state, n_assets, top_k=1)[0][0]
    top_bits = np.array(top_bits)

    # NOTE: QAOA's Ising bitstring convention (bit -> spin) vs the QUBO's
    # x convention are related by x_i = (1 - s_i)/2 = bit_i directly here
    # since we defined bit=0 -> spin+1 -> x=0. So top_bits IS the x vector.
    match = np.array_equal(top_bits, true_x)
    qaoa_bitstring_val = qubo_objective(top_bits, Q, const)
    gap = qaoa_bitstring_val - true_val  # 0 = perfect; small positive = near-miss

    if verbose:
        print(f"\n--- n_assets={n_assets}, budget={budget} ---")
        print(f"Exact optimum (brute force): {true_x}  obj={true_val:.4f}")
        print(f"QAOA top bitstring:          {top_bits}  obj={qaoa_bitstring_val:.4f}  "
              f"(exact match: {match}, gap from true optimum: {gap:.4f})")

    return match, true_val, qaoa_val, gap


def plot_convergence():
    """Track QAOA's cost expectation across optimizer iterations for one run."""
    mu, sigma, names = get_market_data(5, seed=0)
    Q, const, _ = build_qubo(mu, sigma, budget=2, risk_aversion=1.0)
    h, J, offset = qubo_to_ising(Q, const)

    n, p = 5, 2
    cost_diag = build_cost_diagonal(h, J, n)
    history = []

    def objective(params):
        val, _ = qaoa_expectation(params, cost_diag, n, p)
        history.append(val)
        return val

    rng = np.random.default_rng(0)
    x0 = rng.uniform(0, np.pi, size=2 * p)
    scipy_minimize(objective, x0, method="COBYLA", options={"maxiter": 300, "rhobeg": 0.5})

    true_min = cost_diag.min()

    plt.figure(figsize=(7, 4.5))
    plt.plot(history, lw=1.5, label="QAOA expected cost")
    plt.axhline(true_min, color="crimson", ls="--", label="True minimum (exact)")
    plt.xlabel("Optimizer iteration")
    plt.ylabel("Expected cost  ⟨H_C⟩")
    plt.title("QAOA variational convergence (5-asset portfolio)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qaoa_convergence.png", dpi=150)
    plt.close()
    print("Saved qaoa_convergence.png")


def plot_success_vs_n(max_n=9, trials_per_n=5, near_miss_tolerance=0.1):
    """
    Sweep the number of assets (qubits) and measure:
      - exact success rate: how often QAOA's top bitstring matches the
        true optimum exactly
      - near-miss rate: how often QAOA lands within `near_miss_tolerance`
        (relative) of the true optimal objective value, even if it picked
        a different bitstring

    Both matter for the article: exact match answers "did it find THE
    answer", near-miss answers "did it find A good answer". These can
    diverge a lot -- e.g. real market data often has near-tied portfolios
    (two different subsets of similarly-correlated stocks giving nearly
    identical risk/return), so exact match rate can look far worse than
    the practical quality of QAOA's answer.
    """
    ns = list(range(3, max_n + 1))
    success_rates = []
    near_miss_rates = []
    avg_gaps = []

    for n in ns:
        budget = max(1, n // 2)
        successes = 0
        near_misses = 0
        gaps = []
        for trial in range(trials_per_n):
            match, true_val, _, gap = run_single_experiment(
                n, budget, p=2, seed=trial, verbose=False
            )
            successes += int(match)
            gaps.append(gap)
            # Relative tolerance guards against true_val being ~0
            scale = max(abs(true_val), 1e-6)
            if gap / scale <= near_miss_tolerance:
                near_misses += 1

        rate = successes / trials_per_n
        near_miss_rate = near_misses / trials_per_n
        avg_gap = float(np.mean(gaps))

        success_rates.append(rate)
        near_miss_rates.append(near_miss_rate)
        avg_gaps.append(avg_gap)

        print(f"n_assets={n}: exact success rate = {rate:.2f}  "
              f"near-miss rate (<= {near_miss_tolerance:.0%} gap) = {near_miss_rate:.2f}  "
              f"avg gap = {avg_gap:.4f}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(ns, success_rates, marker="o", lw=1.5, color="darkorange", label="Exact match")
    plt.plot(ns, near_miss_rates, marker="s", lw=1.5, color="steelblue",
              label=f"Near-miss (within {near_miss_tolerance:.0%})")
    plt.xlabel("Number of assets (qubits)")
    plt.ylabel("Fraction of trials")
    plt.ylim(-0.05, 1.05)
    plt.title(f"QAOA accuracy vs. problem size (p=2, {trials_per_n} trials/n)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qaoa_vs_n.png", dpi=150)
    plt.close()
    print("Saved qaoa_vs_n.png")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Classical (continuous) mean-variance baseline")
    print("=" * 60)
    mu, sigma, names = get_market_data(5, seed=0)
    print(f"Assets used: {names}")
    w, val = classical_mean_variance(mu, sigma)
    print(f"Optimal continuous weights: {np.round(w, 3)}")
    print(f"Objective value: {val:.4f}")

    print("\n" + "=" * 60)
    print("STEP 2: Single QUBO/QAOA experiment (5 assets, pick 2)")
    print("=" * 60)
    run_single_experiment(n_assets=5, budget=2, p=2, seed=0)

    print("\n" + "=" * 60)
    print("STEP 3: Convergence plot")
    print("=" * 60)
    plot_convergence()

    print("\n" + "=" * 60)
    print("STEP 4: Success rate vs. problem size (this will take ~1-2 min)")
    print("=" * 60)
    plot_success_vs_n(max_n=5, trials_per_n=5)
