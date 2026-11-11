# Anomaly Detection with the Multivariate Normal Distribution

DS5020 Intro Linear Algebra & Probability — Final Project (Topic 9)

We fit a multivariate normal model N(μ, Σ) to a genuine user's keystroke timings
(log hold times and latencies), score new typings with the squared Mahalanobis distance
d² = (x − μ)ᵀ Σ⁻¹ (x − μ), and flag a reading as anomalous when d² exceeds the
(1 − α) quantile of the χ²_d distribution. Everything is implemented from scratch in NumPy.

## Team

| Member | Role |
|---|---|
| Bijun Wu | Implementation and Data Lead |
| Kanchan Patkar | Math and Derivations Lead |

## Repository structure

```
src/       Python code (data loading, simulation, method)
data/      datasets (raw files are NOT committed — see below)
figures/   plots produced by the scripts
docs/      Overleaf report PDF and Excel project plan
```

## Setup

Requires Python 3 with `numpy`, `pandas`, `matplotlib` and `scipy`:

```bash
pip install numpy pandas matplotlib scipy
```

## Running

Simulation check (known true μ, Σ; verifies the Iteration 2 derivations):

```bash
python3 src/simulate.py
```

It prints the checks for each math question (Q1–Q5) and saves figures to `figures/`.
The random seed is fixed (`np.random.default_rng(0)`), so everyone gets the same numbers.

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
