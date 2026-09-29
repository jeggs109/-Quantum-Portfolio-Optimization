"""
qaoa_scratch.py

A from-scratch statevector simulation of QAOA (Quantum Approximate
Optimization Algorithm) -- no Qiskit, just numpy. For n <= ~14 qubits
this is completely tractable (2^14 = 16384-dim state vector), which
covers any reasonably-sized article demo.

THE PHYSICS:
QAOA prepares the state:
    |psi(gamma, beta)> = U_B(beta_p) U_C(gamma_p) ... U_B(beta_1) U_C(gamma_1) |+>^n

- |+>^n : equal superposition over all bitstrings (start every spin
  "undecided", maximum entropy state)
- U_C(gamma) = exp(-i * gamma * H_C) : "cost" unitary, built from the
  Ising Hamiltonian H_C = sum h_i Z_i + sum J_ij Z_i Z_j. This is a
  DIAGONAL operator in the computational basis (a phase rotation per
  bitstring proportional to that bitstring's cost) -- so it's cheap to
  apply even without building a full 2^n x 2^n matrix.
- U_B(beta) = exp(-i * beta * H_B), H_B = sum_i X_i : the "mixer",
  literally a product of single-qubit X-rotations. This is what lets
  amplitude flow between different bitstrings (spin flips), analogous
  to a transverse field driving spin dynamics in a quantum Ising model.

We then classically optimize (gamma, beta) to minimize
<psi | H_C | psi>, i.e. the expected cost -- exactly like variationally
minimizing the energy of a trial wavefunction in physics (Rayleigh-Ritz).
"""

import numpy as np
from scipy.optimize import minimize


def build_cost_diagonal(h, J, n):
    """
    Builds the diagonal of H_C in the computational basis: a length-2^n
    array where entry `idx` is the Ising energy of the bitstring
    represented by `idx` (bit i of idx <-> spin s_i = +1 if bit==0 else -1).

    This avoids ever constructing a dense 2^n x 2^n matrix for H_C.
    """
    dim = 2 ** n
    # spins[:, i] = the value of spin i (+1/-1) for every basis state index
    bits = ((np.arange(dim)[:, None] >> np.arange(n)[None, :]) & 1)
    spins = 1 - 2 * bits  # bit=0 -> spin +1, bit=1 -> spin -1

    diag = spins @ h
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0:
                diag += J[i, j] * spins[:, i] * spins[:, j]
    return diag


def apply_mixer(state, beta, n):
    """
    Applies U_B(beta) = exp(-i * beta * sum_i X_i) = product of
    single-qubit rotations exp(-i * beta * X_i) to the statevector.

    exp(-i*beta*X) = cos(beta) I - i sin(beta) X, applied qubit-by-qubit
    by reshaping the state into a tensor and mixing pairs of amplitudes
    that differ only in bit i.
    """
    dim = len(state)
    cos_b, sin_b = np.cos(beta), np.sin(beta)

    for i in range(n):
        state = state.reshape(-1)
        idx = np.arange(dim)
        bit_i = (idx >> i) & 1
        partner = idx ^ (1 << i)  # flip bit i

        new_state = np.empty_like(state)
        # For bit_i == 0: new = cos*old0 - i*sin*old1 (old1 = partner amplitude)
        # For bit_i == 1: new = cos*old1 - i*sin*old0
        mask0 = bit_i == 0
        new_state[mask0] = cos_b * state[mask0] - 1j * sin_b * state[partner[mask0]]
        new_state[~mask0] = cos_b * state[~mask0] - 1j * sin_b * state[partner[~mask0]]
        state = new_state

    return state


def qaoa_expectation(params, cost_diag, n, p):
    """
    Runs the QAOA circuit for given params = [gamma_1..gamma_p, beta_1..beta_p]
    and returns <H_C> for the resulting state.
    """
    gammas = params[:p]
    betas = params[p:]

    dim = 2 ** n
    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)  # |+>^n

    for layer in range(p):
        # Cost unitary: diagonal phase rotation, cheap elementwise multiply
        state = state * np.exp(-1j * gammas[layer] * cost_diag)
        # Mixer unitary
        state = apply_mixer(state, betas[layer], n)

    probs = np.abs(state) ** 2
    expectation = np.sum(probs * cost_diag)
    return expectation, state


def run_qaoa(h, J, n, p=2, n_restarts=6, seed=0):
    """
    Classically optimizes QAOA parameters to minimize <H_C>, using
    multiple random restarts (the QAOA landscape is non-convex).

    Returns: best_state (statevector), best_params, best_expectation, cost_diag
    """
    rng = np.random.default_rng(seed)
    cost_diag = build_cost_diagonal(h, J, n)

    best_val = np.inf
    best_state = None
    best_params = None

    for r in range(n_restarts):
        x0 = rng.uniform(0, np.pi, size=2 * p)

        def objective(params):
            val, _ = qaoa_expectation(params, cost_diag, n, p)
            return val

        res = minimize(objective, x0, method="COBYLA",
                        options={"maxiter": 300, "rhobeg": 0.5})

        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
            _, best_state = qaoa_expectation(res.x, cost_diag, n, p)

    return best_state, best_params, best_val, cost_diag


def most_likely_bitstrings(state, n, top_k=5):
    """Returns the top_k most probable bitstrings (as tuples) and their probabilities."""
    probs = np.abs(state) ** 2
    top_idx = np.argsort(probs)[::-1][:top_k]
    results = []
    for idx in top_idx:
        bits = tuple((idx >> i) & 1 for i in range(n))
        results.append((bits, probs[idx]))
    return results


if __name__ == "__main__":
    # Small sanity check: 4-qubit random Ising problem
    rng = np.random.default_rng(1)
    n = 4
    h = rng.uniform(-1, 1, n)
    J = np.triu(rng.uniform(-1, 1, (n, n)), k=1)

    state, params, val, cost_diag = run_qaoa(h, J, n, p=3, n_restarts=8)
    print(f"QAOA best expectation value found: {val:.4f}")
    print(f"True minimum (brute force): {cost_diag.min():.4f}")

    print("\nTop bitstrings sampled from QAOA output state:")
    for bits, prob in most_likely_bitstrings(state, n):
        print(f"  {bits}  prob={prob:.3f}")
