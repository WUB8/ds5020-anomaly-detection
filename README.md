# Anomaly Detection with the Multivariate Normal Distribution

DS5020 Intro Linear Algebra & Probability — Final Project (Topic 9)

We fit a multivariate normal model N(μ, Σ) to machine/robot sensor readings from normal
operation, score new readings with the squared Mahalanobis distance
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

The real sensor dataset is not stored in Git. Download instructions will be added here once
the dataset is confirmed; place the raw files in `data/`.
