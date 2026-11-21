# Anomaly Detection with the Multivariate Normal Distribution

DS5020 Intro Linear Algebra & Probability — Final Project (Topic 9)

We fit a multivariate normal model N(μ, Σ) to a genuine user's keystroke timings
(log hold times and latencies), score new typings with the squared Mahalanobis distance
d² = (x − μ)ᵀ Σ⁻¹ (x − μ), and flag a reading as anomalous when d² exceeds the
(1 − α) quantile of the χ²_d distribution. Everything is implemented from scratch in NumPy.

## Team

| Member | Role |
|---|---|
| Bijun Wu (Project Leader)| Implementation and Data Lead |
| Kanchan Patkar | Math and Derivations Lead |

## Repository structure

```
src/       Python code (data loading, simulation, method)
data/      datasets (raw files are NOT committed — see below)
figures/   plots produced by the scripts
docs/      report PDFs, LaTeX sources (report/ = Iteration 2, report_iteration3/ = Iteration 3), Excel project plan
```

## Setup

Requires Python 3 with `numpy`, `pandas`, `matplotlib` and `scipy`:

```bash
pip install numpy pandas matplotlib scipy
```

## Code (Iteration 3)

| File | What it is |
|---|---|
| `src/methods.py` | The method from scratch (NumPy only, no `numpy.linalg`/SciPy). One `my_` function per algorithm step, with its equation in a comment: `my_mle`, `my_jacobi_eig`, `my_inv_sqrt`, `my_whiten_distance2`, `my_ldl`, `my_cholesky`, `my_forward_sub`, `my_mahalanobis2`, `my_inverse`, `my_threshold` (χ² quantile), `my_threshold_F` (small-sample F threshold), `my_decide`, `my_rates` |
| `src/simulate.py` | Iteration 2 simulation model as functions: `simulate_normal(d, n, seed)`, `simulate_t(...)` (stress test), `add_duplicate_feature(X)` |
| `src/diagnostics.py` | Checks only: `report()` comparison helper, verification-table rows, assumption checks, Q-Q plots |
| `src/load.py` | Load and describe the real data |
| `src/iteration2_check.py` | The Iteration 2 simulation script, unchanged, to reproduce the Iteration 2 numbers |

`numpy.linalg`, SciPy and scikit-learn are used only in the scripts below, to check our answers.

## Running

All scripts run from the project root. Seeds are fixed, so everyone gets the same numbers; figures go to `figures/`.

The tiny example step by step (4 points in R², exact fractions; Iteration 3 Q1, Q2):

```bash
python3 src/small_example.py
```

The simulation test (truth and library comparisons, error against n, stress tests, assumption checks on simulated data):

```bash
python3 src/check_simulation.py
```

The real data and the library comparison (needs the data file, see below):

```bash
python3 src/run_real.py
```

Each script ends with the rows it contributes to the verification table in the report.

## Data

We use the **CMU Keystroke Dynamics Benchmark** (Killourhy and Maxion, 2009): 51 subjects each
typed the password `.tie5Roanl` 400 times; each row has 31 timing features (hold times and
key-to-key latencies). Source: https://www.cs.cmu.edu/~keystroke/

The raw file is not stored in Git. Download it into `data/`:

```bash
curl -L -o data/DSL-StrongPasswordData.csv https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv
```

Then load and describe it (size, types, missing values, summary statistics, two plots in `figures/`):

```bash
python3 src/load.py
```
