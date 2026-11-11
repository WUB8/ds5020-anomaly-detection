"""
DS5020 Final Project - Iteration 2: Simulation Check
Project 9: Anomaly Detection with the Multivariate Normal Distribution

Part A: a tiny 2-D example with round numbers that you can redo by hand.
Part B: the main simulation (d = 5, n = 5000) where the true mu, Sigma are known.
Part C: heavy-tailed data (t with 3 df) to show what breaks when normality fails.

Run from the project root:  python3 src/simulate.py
Figures are saved to figures/.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# Core method (written from scratch, numpy only)
# ---------------------------------------------------------------------------
def fit_mle(X):
    """Q1: MLEs of a multivariate normal. Note 1/n (MLE), not 1/(n-1)."""
    n = X.shape[0]
    mu_hat = X.mean(axis=0)                 # (1/n) X^T 1
    Xc = X - mu_hat
    S_hat = Xc.T @ Xc / n                   # (1/n) Xc^T Xc
    return mu_hat, S_hat


def inv_sqrt(S):
    """Q4: S^{-1/2} = Q Lambda^{-1/2} Q^T from the eigendecomposition S = Q Lambda Q^T."""
    lam, Q = np.linalg.eigh(S)
    if lam.min() <= 0:
        raise ValueError("Sigma is not positive definite")
    return Q @ np.diag(lam ** -0.5) @ Q.T, lam, Q


def mahalanobis_sq(X, mu, S_inv_half):
    """d^2 = (x - mu)^T S^{-1} (x - mu) = ||S^{-1/2}(x - mu)||^2."""
    Z = (X - mu) @ S_inv_half              # S^{-1/2} is symmetric, so this is fine
    return np.sum(Z ** 2, axis=1), Z


def log_lik(X, mu, S):
    """Q1: l(mu, S) = -n d/2 log(2 pi) - n/2 log|S| - 1/2 sum (x_i-mu)^T S^{-1} (x_i-mu)."""
    n, d = X.shape
    Xc = X - mu
    _, logdet = np.linalg.slogdet(S)
    quad = np.sum(Xc @ np.linalg.inv(S) * Xc)
    return -0.5 * n * d * np.log(2 * np.pi) - 0.5 * n * logdet - 0.5 * quad


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Part A: hand-worked 2-D example
# ---------------------------------------------------------------------------
header("PART A: hand-worked example (d = 2, n = 4)")
Xh = np.array([[1.0, 2.0],
               [3.0, 4.0],
               [5.0, 4.0],
               [3.0, 6.0]])
mu_h, S_h = fit_mle(Xh)
print("data X =\n", Xh)
print("mu_hat            =", mu_h, "   (expect [3, 4])")
print("centred Xc =\n", Xh - mu_h)
print("Sigma_hat =\n", S_h, "   (expect [[2, 1], [1, 2]])")

Sih_h, lam_h, Q_h = inv_sqrt(S_h)
print("eigenvalues       =", lam_h, "   (expect 1 and 3)")
print("eigenvectors Q =\n", Q_h, "   (expect (1,-1)/sqrt2 and (1,1)/sqrt2, up to sign)")
print("Sigma_hat^{-1} =\n", np.linalg.inv(S_h), "   (expect (1/3)[[2,-1],[-1,2]])")
print("Sigma_hat^{-1/2} =\n", Sih_h)
print("check (S^{-1/2})^2 = S^{-1}:", np.allclose(Sih_h @ Sih_h, np.linalg.inv(S_h)))

tests = np.array([[5.0, 6.0], [5.0, 2.0]])
d2_h, _ = mahalanobis_sq(tests, mu_h, Sih_h)
eucl = np.linalg.norm(tests - mu_h, axis=1)
for x, e, m in zip(tests, eucl, d2_h):
    print(f"x = {x}:  Euclidean dist = {e:.4f},  Mahalanobis d^2 = {m:.4f}")
print("   (expect d^2 = 8/3 = 2.6667 and 8; same Euclidean distance sqrt(8))")
for a in (0.05, 0.01):
    tau = stats.chi2.ppf(1 - a, df=2)
    print(f"alpha = {a}: tau = chi2_2 quantile = {tau:.4f}  -> flagged: {d2_h > tau}")


# ---------------------------------------------------------------------------
# Part B: main simulation with known truth
# ---------------------------------------------------------------------------
header("PART B: main simulation (d = 5, n = 5000, alpha = 0.01)")
rng = np.random.default_rng(0)
d, n, alpha = 5, 5000, 0.01
B = rng.normal(size=(d, d))
Sigma = B @ B.T + d * np.eye(d)            # symmetric positive definite by construction
mu = np.zeros(d)
X = rng.multivariate_normal(mu, Sigma, size=n)

# Q1: MLEs
mu_hat, S_hat = fit_mle(X)
print("\n[Q1] MLE vs truth")
print(f"{'param':<14}{'true':>10}{'estimate':>12}{'abs err':>10}")
for i in range(d):
    print(f"{'mu_' + str(i + 1):<14}{mu[i]:>10.4f}{mu_hat[i]:>12.4f}{abs(mu_hat[i] - mu[i]):>10.4f}")
for i in range(d):
    print(f"{'Sigma_' + f'{i + 1}{i + 1}':<14}{Sigma[i, i]:>10.4f}{S_hat[i, i]:>12.4f}"
          f"{abs(S_hat[i, i] - Sigma[i, i]):>10.4f}")
rel = np.linalg.norm(S_hat - Sigma) / np.linalg.norm(Sigma)
print(f"relative Frobenius error ||S_hat - Sigma|| / ||Sigma|| = {rel:.4f}")
print("MLE (1/n) vs np.cov (1/(n-1)) ratio:", np.round((S_hat / np.cov(X.T)).mean(), 6),
      f"(expect (n-1)/n = {(n - 1) / n:.6f})")

# Q1: the MLE should maximise the log-likelihood -> any perturbation lowers it
ll_star = log_lik(X, mu_hat, S_hat)
worse = []
for _ in range(20):
    dm = 0.05 * rng.normal(size=d)
    E = 0.05 * rng.normal(size=(d, d)); E = E @ E.T
    worse.append(log_lik(X, mu_hat + dm, S_hat + E) < ll_star)
print(f"log-lik at MLE = {ll_star:.2f}; 20 random perturbations all lower: {all(worse)}")

# Q2: Mahalanobis distance non-negative when Sigma is PD
S_inv_half, lam, Q = inv_sqrt(S_hat)
d2, Z = mahalanobis_sq(X, mu_hat, S_inv_half)
print("\n[Q2] eigenvalues of Sigma_hat (all > 0 => PD):", lam)
print(f"     min d^2 = {d2.min():.6f}  (>= 0: {bool(d2.min() >= 0)})")
d2_direct = np.einsum("ij,jk,ik->i", X - mu_hat, np.linalg.inv(S_hat), X - mu_hat)
print(f"     d^2 via Sigma^-1 equals d^2 via ||Z||^2: {np.allclose(d2, d2_direct)}")

# Q4: Sigma^{-1/2} from the eigendecomposition; whitening
print("\n[Q4] Q^T Q = I:", np.allclose(Q.T @ Q, np.eye(d)),
      "| Q Lambda Q^T = Sigma_hat:", np.allclose(Q @ np.diag(lam) @ Q.T, S_hat))
print("     S^{-1/2} symmetric:", np.allclose(S_inv_half, S_inv_half.T),
      "| S^{-1/2} S S^{-1/2} = I:", np.allclose(S_inv_half @ S_hat @ S_inv_half, np.eye(d)))
print("     cov(Z) (should be identity):\n", np.cov(Z.T, bias=True).round(4))

# Q3: d^2 ~ chi^2_d
print(f"\n[Q3] mean of d^2 = {d2.mean():.4f} (chi2_{d} mean = {d})")
print(f"     var  of d^2 = {d2.var():.4f} (chi2_{d} var  = {2 * d})")
ks = stats.kstest(d2, "chi2", args=(d,))
print(f"     KS test vs chi2_{d}: statistic = {ks.statistic:.4f}, p-value = {ks.pvalue:.4f}")

# Q5: false-alarm rate at the (1 - alpha) quantile
print("\n[Q5] false-alarm rate on normal data")
print(f"{'alpha':>8}{'tau':>10}{'observed':>12}")
for a in (0.10, 0.05, 0.01):
    tau_a = stats.chi2.ppf(1 - a, df=d)
    print(f"{a:>8.2f}{tau_a:>10.4f}{np.mean(d2 > tau_a):>12.4f}")
tau = stats.chi2.ppf(1 - alpha, df=d)

# Detection of shifted "attacks"
attacks = rng.multivariate_normal(mu + 4, Sigma, size=50)
d2a, _ = mahalanobis_sq(attacks, mu_hat, S_inv_half)
print(f"\n     detection rate on 50 attacks (mean shifted by +4): {np.mean(d2a > tau):.2f}")

# Figure: histogram of d^2 vs chi^2_d density
grid = np.linspace(0, max(d2.max(), d2a.max()) * 1.02, 400)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(d2, bins=80, density=True, alpha=0.6, label=r"simulated $d^2$ (normal data)")
ax.plot(grid, stats.chi2.pdf(grid, df=d), "k-", lw=2, label=rf"$\chi^2_{d}$ density")
ax.axvline(tau, color="red", ls="--", label=rf"threshold $\tau$ ($\alpha$={alpha})")
ax.set_xlim(0, 40)
ax.set_xlabel(r"$d^2$"); ax.set_ylabel("density"); ax.legend()
ax.set_title(r"Q3: squared Mahalanobis distance vs $\chi^2_5$")
fig.tight_layout(); fig.savefig(FIG / "sim_d2_hist.png", dpi=150); plt.close(fig)

# Figure: Q-Q plot of d^2 vs chi^2_d
probs = (np.arange(1, n + 1) - 0.5) / n
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(stats.chi2.ppf(probs, df=d), np.sort(d2), ".", ms=2)
lim = stats.chi2.ppf(probs[-1], df=d)
ax.plot([0, lim], [0, lim], "r-")
ax.set_xlabel(rf"$\chi^2_{d}$ quantiles"); ax.set_ylabel(r"sorted $d^2$")
ax.set_title("Q-Q plot (normal data)")
fig.tight_layout(); fig.savefig(FIG / "sim_d2_qq.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Part C: heavy tails break the chi^2 threshold
# ---------------------------------------------------------------------------
header("PART C: heavy-tailed data (multivariate t, 3 df)")
nu = 3
W = rng.chisquare(nu, size=n)
Xt = mu + rng.multivariate_normal(np.zeros(d), Sigma, size=n) / np.sqrt(W / nu)[:, None]
mu_t, S_t = fit_mle(Xt)
Sih_t, _, _ = inv_sqrt(S_t)
d2t, _ = mahalanobis_sq(Xt, mu_t, Sih_t)
print(f"{'alpha':>8}{'normal data':>14}{'t_3 data':>12}")
for a in (0.10, 0.05, 0.01):
    tau_a = stats.chi2.ppf(1 - a, df=d)
    print(f"{a:>8.2f}{np.mean(d2 > tau_a):>14.4f}{np.mean(d2t > tau_a):>12.4f}")
print("-> with heavy tails the false-alarm rate is no longer alpha")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(d2t[d2t < 40], bins=80, density=True, alpha=0.6, label=r"$d^2$ ($t_3$ data)")
ax.plot(grid, stats.chi2.pdf(grid, df=d), "k-", lw=2, label=rf"$\chi^2_{d}$ density")
ax.axvline(tau, color="red", ls="--", label=rf"threshold $\tau$")
ax.set_xlim(0, 40); ax.set_xlabel(r"$d^2$"); ax.set_ylabel("density"); ax.legend()
ax.set_title(r"Heavy tails: $d^2$ no longer follows $\chi^2_5$")
fig.tight_layout(); fig.savefig(FIG / "sim_d2_heavy_tail.png", dpi=150); plt.close(fig)

print(f"\nFigures saved to {FIG}")
