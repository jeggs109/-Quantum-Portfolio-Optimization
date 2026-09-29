"""
qubo_portfolio.py

Converts portfolio SELECTION (a simplified, binary version of portfolio
optimization) into a QUBO (Quadratic Unconstrained Binary Optimization),
and then into an Ising Hamiltonian -- the exact same object physicists
write down for a spin-glass / spin system with pairwise couplings.

THE FINANCE PROBLEM (simplified, cardinality-constrained):
Given N assets, choose a binary vector x in {0,1}^N (x_i = 1 means
"include asset i in the portfolio, weighted equally among chosen assets")
that maximizes return while minimizing risk, subject to picking exactly
B assets.

    minimize:   q * x^T Sigma x  -  mu^T x
    subject to: sum(x_i) = B

This is the same structure as the Qiskit finance textbook's portfolio
optimization QUBO. We fold the budget constraint in as a quadratic
penalty (standard QUBO trick):

    minimize: q * x^T Sigma x - mu^T x + P * (sum(x_i) - B)^2

THE PHYSICS BRIDGE:
Binary variables x_i in {0,1} map to spins s_i in {-1, +1} via
x_i = (1 - s_i) / 2. Substituting turns the QUBO into:

    H(s) = sum_i h_i * s_i + sum_{i<j} J_ij * s_i * s_j + constant

which is literally the Ising Hamiltonian for a system of N spins with
external fields h_i and pairwise couplings J_ij. "Find the best
portfolio" and "find the ground state of a spin glass" are the same
optimization problem.
"""

import numpy as np
from itertools import product


def build_qubo(mu, sigma, budget, risk_aversion=1.0, penalty=None):
    """
    Builds the QUBO matrix Q such that the objective is x^T Q x
    (x_i in {0,1}), for:
        minimize  q * x^T Sigma x - mu^T x + P * (sum(x) - B)^2

    Returns Q (n x n matrix) and a constant offset (doesn't affect
    the argmin, only the objective value).
    """
    n = len(mu)
    if penalty is None:
        # Heuristic: penalty should dominate the scale of the objective
        # so constraint violations are never favorable.
        penalty = 2.0 * (risk_aversion * np.max(np.abs(sigma)) + np.max(np.abs(mu)))

    Q = risk_aversion * sigma.copy()

    # Linear return term: -mu^T x contributes to diagonal
    np.fill_diagonal(Q, np.diag(Q) - mu)

    # Budget penalty: P*(sum(x) - B)^2 = P*sum(x_i x_j) - 2*P*B*sum(x_i) + P*B^2
    Q += penalty * np.ones((n, n))
    np.fill_diagonal(Q, np.diag(Q) - 2 * penalty * budget)
    constant = penalty * budget ** 2

    return Q, constant, penalty


def qubo_to_ising(Q, constant):
    """
    Converts QUBO matrix Q (objective x^T Q x) into Ising form:
        H(s) = offset + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j
    using x_i = (1 - s_i) / 2.

    Returns h (n,), J (n x n, upper triangular, zero diagonal), offset (float)
    """
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = constant

    # Symmetrize Q first so cross terms are counted once
    Qs = (Q + Q.T) / 2

    for i in range(n):
        offset += Qs[i, i] / 2
        h[i] -= Qs[i, i] / 2
        for j in range(n):
            if i == j:
                continue
            offset += Qs[i, j] / 4
            h[i] -= Qs[i, j] / 2
            if j > i:
                J[i, j] += Qs[i, j] / 2

    return h, J, offset


def qubo_objective(x, Q, constant):
    """Evaluate x^T Q x + constant for a binary vector x."""
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x + constant)


def brute_force_solve(Q, constant):
    """
    Exhaustively try every binary vector (fine for n <= ~20).
    Returns (best_x, best_value) -- the EXACT ground truth we'll
    compare QAOA against.
    """
    n = Q.shape[0]
    best_x, best_val = None, np.inf
    for bits in product([0, 1], repeat=n):
        val = qubo_objective(bits, Q, constant)
        if val < best_val:
            best_val = val
            best_x = bits
    return np.array(best_x), best_val


if __name__ == "__main__":
    from synthetic_market import generate_synthetic_returns, compute_mu_sigma

    returns = generate_synthetic_returns(n_assets=5, n_days=500)
    mu, sigma = compute_mu_sigma(returns)

    Q, const, penalty = build_qubo(mu, sigma, budget=2, risk_aversion=1.0)
    h, J, offset = qubo_to_ising(Q, const)

    print(f"Penalty strength used: {penalty:.4f}")
    print(f"Ising field terms h: {np.round(h, 3)}")
    print(f"Ising coupling matrix J:\n{np.round(J, 3)}")

    best_x, best_val = brute_force_solve(Q, const)
    print(f"\nExact best selection (brute force): {best_x}, objective={best_val:.4f}")
