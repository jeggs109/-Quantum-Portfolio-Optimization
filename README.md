# Quantum Portfolio Optimization, From Scratch

A self-contained (numpy/scipy/matplotlib only, no Qiskit) simulation showing
how portfolio selection maps onto an Ising spin model, and how QAOA
(a quantum optimization algorithm) performs against the exact classical
answer as the problem grows.

## Run it

```bash
pip install numpy scipy matplotlib
python3 main.py
```

Takes about 1-2 minutes. Produces two plots:
- `qaoa_convergence.png` — QAOA's cost estimate converging during
  classical optimization of its parameters
- `qaoa_vs_n.png` — success rate (finding the true optimal portfolio)
  vs. number of assets

## Files

| File | What it does |
|---|---|
| `synthetic_market.py` | Generates correlated synthetic stock returns (Cholesky decomposition on correlated Gaussian noise) |
| `qubo_portfolio.py` | Formulates portfolio selection as a QUBO, converts to an Ising Hamiltonian, brute-force exact solver |
| `qaoa_scratch.py` | QAOA implemented as a raw numpy statevector simulator — the actual quantum circuit, not a library call |
| `main.py` | Runs the full pipeline: classical baseline → QUBO → QAOA → comparison plots |

## The physics-finance bridge (the article's spine)

1. **The finance problem**: pick a subset of B assets out of N to
   minimize risk and maximize return. This is naturally a discrete,
   combinatorial optimization — not the smooth continuous problem
   Markowitz originally solved.

2. **The QUBO reformulation**: any problem of the form "minimize a
   quadratic function of binary choices, subject to a budget
   constraint" can be written as `x^T Q x`. The constraint gets folded
   in as a penalty term. This is a completely mechanical, standard
   trick (see `qubo_portfolio.py::build_qubo`).

3. **The physics**: substituting `x_i = (1-s_i)/2` turns the QUBO into
   an **Ising Hamiltonian** — literally the same object physicists
   write for a system of interacting spins with an external field and
   pairwise couplings. "Find the best portfolio" and "find the
   spin-glass ground state" are now *the same equation*.

4. **QAOA**: alternates a "cost" unitary (a phase rotation encoding the
   Ising energy of each spin configuration) with a "mixer" unitary
   (transverse-field-style X-rotations that let amplitude move between
   configurations). This is the quantum-computing analogue of quantum
   annealing. The circuit's few parameters are tuned classically to
   minimize the expected energy — a variational method exactly like
   minimizing `<psi|H|psi>` for a trial wavefunction in physics.

## The honest finding (don't skip this in the article)

At shallow circuit depth (p=2) with a small number of classical
optimizer restarts, QAOA's success rate at finding the *exact* optimal
portfolio collapses quickly as the number of assets grows — in this
run, from ~40% success at 3 assets to ~0% by 5-6 assets. This is the
real, well-documented behavior of QAOA on small/noiseless simulators:
shallow circuits and non-convex classical optimization landscapes make
it easy to get stuck in local optima long before you'd need actual
quantum hardware to run out of steam.

This is *the* interesting point for an econophysics-style article:
the quantum reformulation is mathematically clean and physically
meaningful (portfolio selection really is a spin-glass problem), but
turning that into a practical computational advantage is still an open
research problem — which matches what banks like JPMorgan and Goldman
say publicly about their own quantum finance pilots (research-stage,
not production).

## Extending this for the article

- **Real data**: replace `generate_synthetic_returns()` with `yfinance`
  price history, computing daily % returns the same way — nothing
  downstream changes.
- **Deeper QAOA (p > 2)**: increase `p` in `run_qaoa()`. Success rate
  should improve, at the cost of a harder classical optimization
  landscape — a good second plot.
- **Real quantum hardware / Qiskit**: once you have internet/package
  access, IBM's `qiskit-optimization` finance module (`PortfolioOptimization`
  class) implements the identical QUBO formulation used here — you can
  validate this from-scratch code against it, then run on real IBM
  hardware via their free tier.
- **D-Wave**: the Ising form (`h`, `J` from `qubo_to_ising`) is exactly
  the input format D-Wave's quantum annealers expect, if you want to
  compare gate-based QAOA against annealing on the same problem.
