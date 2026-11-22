"""
DS5020 Final Project - Iteration 3: simulation model as functions
Project 9: Anomaly Detection with the Multivariate Normal Distribution

Same model, parameter values and seed as Iteration 2 (Part B):
    B ~ N(0, 1) entries (d x d),  Sigma = B B^T + d I,  mu = 0,
    x_i ~ N(mu, Sigma) iid,  rng = np.random.default_rng(seed), seed = 0.
simulate_normal(d=5, n=5000, seed=0) returns exactly the Iteration 2 data X.

The random draws are made row by row, so for a fixed seed the first n rows are the
same for every sample size: simulate_normal(n=500)[0] == simulate_normal(n=5000)[0][:500].
The truth (mu, Sigma) depends only on d and the seed, so it is the same for every n.
"""
import numpy as np


def _truth(rng, d):
    B = rng.normal(size=(d, d))
    Sigma = B @ B.T + d * np.eye(d)          # symmetric positive definite by construction
    mu = np.zeros(d)
    return mu, Sigma


def simulate_normal(d=5, n=5000, seed=0, rep=None):
    """Iteration 2 model. Returns X (n x d), the true mu and the true Sigma.
    rep=None: data drawn from the same generator as the truth (exactly Iteration 2).
    rep=r:    same truth (from seed), new independent data set number r, for repeated
              experiments with a fixed mu and Sigma."""
    rng = np.random.default_rng(seed)
    mu, Sigma = _truth(rng, d)
    if rep is not None:
        rng = np.random.default_rng([seed, rep])
    X = rng.multivariate_normal(mu, Sigma, size=n)
    return X, mu, Sigma


def simulate_t(d=5, n=5000, nu=3, seed=0):
    """Stress test (Assumption 2 broken): multivariate t with nu degrees of freedom,
    x = mu + y / sqrt(w / nu),  y ~ N(0, Sigma),  w ~ chi^2_nu.
    Same mu, Sigma (scale matrix) as simulate_normal for the same d and seed.
    Note Cov(x) = nu / (nu - 2) * Sigma for nu > 2."""
    rng = np.random.default_rng(seed)
    mu, Sigma = _truth(rng, d)
    Y = rng.multivariate_normal(np.zeros(d), Sigma, size=n)
    w = rng.chisquare(nu, size=n)
    X = mu + Y / np.sqrt(w / nu)[:, None]
    return X, mu, Sigma


def add_duplicate_feature(X, j=0):
    """Stress test (Assumption 3 broken): append an exact copy of feature j,
    so the columns of X_c are linearly dependent and Sigma_hat is singular."""
    return np.column_stack([X, X[:, j]])


if __name__ == "__main__":
    X, mu, Sigma = simulate_normal()
    print("simulate_normal(d=5, n=5000, seed=0):", X.shape)
    print("true mu =", mu)
    print("true Sigma =\n", Sigma.round(4))
    X500, _, _ = simulate_normal(n=500)
    print("first 500 rows identical to n=500 run:", np.array_equal(X[:500], X500))
