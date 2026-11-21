"""
DS5020 Final Project - Iteration 3: simulation test (recover the known truth)
Project 9: Anomaly Detection with the Multivariate Normal Distribution

Same model, parameters and seed as Iteration 2: d = 5, mu = 0, Sigma = B B^T + 5 I,
n = 5000, alpha = 0.01, seed 0 (src/simulate.py).

  Part A  from-scratch functions vs library (numpy.linalg / scipy): rounding level
  Part B  from-scratch estimates vs the truth: within 3 standard errors
  Part C  error against n on log-log axes: slope about -1/2
  Part D  false-alarm rate and Q-Q plot of d^2 (Iteration 3, Q3)
  Part E  stress tests: heavy tails (t_3), duplicated feature (singular Sigma_hat),
          small sample n = 200, d = 21 (chi^2 vs F threshold)
  Part F  assumption checks on simulated data (they should pass)

Run from the project root:  python3 src/check_simulation.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from methods import (my_mle, my_cholesky, my_mahalanobis2, my_jacobi_eig, my_inv_sqrt,
                     my_whiten_distance2, my_inverse, my_ldl, my_threshold, my_threshold_F,
                     my_rates)
from simulate import simulate_normal, simulate_t, add_duplicate_feature
from diagnostics import (report, table_row, print_table, header, qq_chi2, assumption_checks,
                         qq_normal_grid, FIG)

np.set_printoptions(precision=4, suppress=True)
d, n, alpha, seed = 5, 5000, 0.01, 0

# ---------------------------------------------------------------------------
# Part A: our functions vs the library on the Iteration 2 data
# ---------------------------------------------------------------------------
header(f"PART A: from-scratch vs library (d = {d}, n = {n}, seed = {seed})")
X, mu, Sigma = simulate_normal(d=d, n=n, seed=seed)       # identical to Iteration 2
mu_hat, S_hat = my_mle(X)                                 # divide by n (MLE)
L = my_cholesky(S_hat)                                    # S_hat = L L^T
d2 = my_mahalanobis2(X, mu_hat, L)                        # solve L z = x - mu_hat
lam, Q = my_jacobi_eig(S_hat)                             # S_hat = Q Lambda Q^T
W = my_inv_sqrt(lam, Q)                                   # W = Q Lambda^{-1/2} Q^T
d2_W = my_whiten_distance2(X, mu_hat, W)                  # z = W v

diff = X - mu_hat
d2_lib = np.sum(diff @ np.linalg.inv(S_hat) * diff, axis=1)
lam_lib, Q_lib = np.linalg.eigh(S_hat)
sgn = np.sign(np.sum(Q * Q_lib, axis=0))                  # eigenvectors: align signs
tau = my_threshold(alpha, d)
tau_lib = stats.chi2.ppf(1 - alpha, d)

rows = [
    ("mu_hat vs X.mean", mu_hat, X.mean(axis=0)),
    ("Sigma_hat vs np.cov(bias=True)", S_hat, np.cov(X.T, bias=True)),
    ("Cholesky vs numpy", L, np.linalg.cholesky(S_hat)),
    ("inverse (LDL) vs numpy.inv", my_inverse(S_hat), np.linalg.inv(S_hat)),
    ("eigenvalues (Jacobi) vs eigh", lam, lam_lib),
    ("eigenvectors vs eigh (signs aligned)", Q * sgn, Q_lib),
    ("d^2 (Cholesky) vs numpy.inv", d2, d2_lib),
    ("d^2 (W) vs numpy.inv", d2_W, d2_lib),
    ("tau = chi2_{5,0.99} vs scipy", tau, tau_lib),
]
for name, mine, ref in rows:
    a, r = report(name, mine, ref)
    if name.startswith(("Cholesky", "inverse", "eigenvalues", "d^2 (Cholesky)", "d^2 (W", "tau")):
        table_row(f"Simulation (n = {n})", name.split(" vs ")[0], name.split(" vs ")[-1],
                  f"max = {a:.0e}", a < 1e-8)
print(f"tau ours = {tau:.10f}, scipy = {tau_lib:.10f}")

# ---------------------------------------------------------------------------
# Part B: estimates vs the truth, in units of the standard error
#   SE(mu_hat_j) = sqrt(Sigma_jj / n)
#   SE(Sigma_hat_jk) ~ sqrt((Sigma_jk^2 + Sigma_jj Sigma_kk) / n)   (normal data)
# ---------------------------------------------------------------------------
header("PART B: estimates vs the known truth (within 3 standard errors?)")
se_mu = np.sqrt(np.diag(Sigma) / n)
z_mu = (mu_hat - mu) / se_mu
se_S = np.sqrt((Sigma ** 2 + np.outer(np.diag(Sigma), np.diag(Sigma))) / n)
z_S = (S_hat - Sigma) / se_S
iu = np.triu_indices(d)
print(f"{'param':<12}{'true':>10}{'estimate':>11}{'SE':>9}{'z':>8}")
for j in range(d):
    print(f"{'mu_' + str(j + 1):<12}{mu[j]:>10.4f}{mu_hat[j]:>11.4f}{se_mu[j]:>9.4f}{z_mu[j]:>8.2f}")
for j in range(d):
    print(f"{f'Sigma_{j + 1}{j + 1}':<12}{Sigma[j, j]:>10.4f}{S_hat[j, j]:>11.4f}"
          f"{se_S[j, j]:>9.4f}{z_S[j, j]:>8.2f}")
err_mu = np.linalg.norm(mu_hat - mu)
rel_S = np.linalg.norm(S_hat - Sigma) / np.linalg.norm(Sigma)
print(f"max |z| over mu: {np.abs(z_mu).max():.2f};  over the {len(iu[0])} entries of Sigma: "
      f"{np.abs(z_S[iu]).max():.2f}")
print(f"||mu_hat - mu||_2 = {err_mu:.4f}  (expected size sqrt(tr Sigma / n) = "
      f"{np.sqrt(np.trace(Sigma) / n):.4f})")
print(f"||Sigma_hat - Sigma||_F / ||Sigma||_F = {rel_S:.4f}")
table_row(f"Simulation (n = {n})", "mu_hat", "true mu",
          f"||.||_2 = {err_mu:.3f}, max|z| = {np.abs(z_mu).max():.1f}", np.abs(z_mu).max() < 3)
table_row(f"Simulation (n = {n})", "Sigma_hat", "true Sigma",
          f"rel F = {rel_S:.3f}, max|z| = {np.abs(z_S[iu]).max():.1f}", np.abs(z_S[iu]).max() < 3)

# ---------------------------------------------------------------------------
# Part C: error against n (same truth, 200 independent data sets per n)
# ---------------------------------------------------------------------------
header("PART C: error against n (log-log), 200 repetitions per n")
n_list = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])
R = 200
e_mu = np.zeros((len(n_list), R))
e_S = np.zeros((len(n_list), R))
for r in range(R):
    Xr, _, _ = simulate_normal(d=d, n=n_list.max(), seed=seed, rep=r)
    for k, nk in enumerate(n_list):                       # nested: first nk rows
        m_k, S_k = my_mle(Xr[:nk])
        e_mu[k, r] = np.linalg.norm(m_k - mu)
        e_S[k, r] = np.linalg.norm(S_k - Sigma) / np.linalg.norm(Sigma)
mean_mu, mean_S = e_mu.mean(axis=1), e_S.mean(axis=1)
slope_mu = np.polyfit(np.log(n_list), np.log(mean_mu), 1)[0]
slope_S = np.polyfit(np.log(n_list), np.log(mean_S), 1)[0]
print(f"{'n':>7}{'mean ||mu_hat-mu||':>20}{'sqrt(tr Sigma/n)':>18}{'mean rel err Sigma':>20}")
for k, nk in enumerate(n_list):
    print(f"{nk:>7}{mean_mu[k]:>20.4f}{np.sqrt(np.trace(Sigma) / nk):>18.4f}{mean_S[k]:>20.4f}")
print(f"log-log slope: mu {slope_mu:.3f}, Sigma {slope_S:.3f}  (theory -0.5)")
table_row("Simulation (n = 50..10^4)", "error vs n", "slope -1/2",
          f"mu {slope_mu:.2f}, Sigma {slope_S:.2f}",
          abs(slope_mu + 0.5) < 0.05 and abs(slope_S + 0.5) < 0.05)

fig, ax = plt.subplots(figsize=(6, 4.2))
ax.loglog(n_list, mean_mu, "o-", label=r"mean $\|\hat\mu-\mu\|_2$")
ax.loglog(n_list, mean_S, "s-", label=r"mean $\|\hat\Sigma-\Sigma\|_F/\|\Sigma\|_F$")
ax.loglog(n_list, np.sqrt(np.trace(Sigma) / n_list), "k--", lw=1,
          label=r"$\sqrt{\mathrm{tr}\,\Sigma/n}$ (slope $-1/2$)")
ax.set_xlabel("sample size $n$"); ax.set_ylabel("error against the truth")
ax.set_title(f"Error vs n ({R} repetitions per n); slopes {slope_mu:.2f}, {slope_S:.2f}")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "sim_error_vs_n.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------------------
# Part D: false-alarm rate and Q-Q plot (Q3)
# ---------------------------------------------------------------------------
header("PART D: false-alarm rate and d^2 vs chi^2_5")
X2, _, _ = simulate_normal(d=d, n=2 * n, seed=seed)       # first n rows = X
X_test = X2[n:]                                           # 5000 new normal points
d2_test = my_mahalanobis2(X_test, mu_hat, L)
print(f"{'alpha':>7}{'tau':>10}{'training':>10}{'held-out':>10}{'3 SE band':>18}")
for a in (0.10, 0.05, 0.01):
    t_a = my_threshold(a, d)
    se = np.sqrt(a * (1 - a) / n)
    far_tr, far_te = np.mean(d2 > t_a), np.mean(d2_test > t_a)
    print(f"{a:>7.2f}{t_a:>10.4f}{far_tr:>10.4f}{far_te:>10.4f}   [{a - 3 * se:.4f}, {a + 3 * se:.4f}]")
far01 = np.mean(d2 > tau)
ks = stats.kstest(d2, "chi2", args=(d,))
print(f"KS test of training d^2 vs chi^2_5: p = {ks.pvalue:.3f}")
table_row(f"Simulation (n = {n})", "false-alarm rate", "alpha = 0.01",
          f"{far01:.4f} (band 0.010 +- 0.004)", abs(far01 - 0.01) <= 0.004)

fig, ax = plt.subplots(figsize=(5, 5))
qq_chi2(ax, d2_test, d, "held-out normal data")
ax.set_title(r"Simulation: held-out $d^2$ vs $\chi^2_5$"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "sim_iter3_d2_qq.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------------------
# Part E1: stress test - heavy tails (Assumption 2 broken)
# ---------------------------------------------------------------------------
header("PART E1: stress test - multivariate t_3 data (Assumption 2 broken)")
Xt, _, _ = simulate_t(d=d, n=2 * n, nu=3, seed=seed)
mt, St = my_mle(Xt[:n])
Lt = my_cholesky(St)
d2t = my_mahalanobis2(Xt[n:], mt, Lt)
print(f"{'alpha':>7}{'normal':>10}{'t_3':>10}")
for a in (0.10, 0.05, 0.01):
    t_a = my_threshold(a, d)
    print(f"{a:>7.2f}{np.mean(d2_test > t_a):>10.4f}{np.mean(d2t > t_a):>10.4f}")
far_t = np.mean(d2t > tau)
print(f"-> at alpha = 0.01 the false-alarm rate is {far_t:.3f}, {far_t / 0.01:.1f} times alpha")
table_row("Simulation (stress: t_3)", "false-alarm rate", "alpha = 0.01",
          f"{far_t:.3f} (fails, as expected)", far_t > 0.01 + 0.004)
fig, ax = plt.subplots(figsize=(5, 5))
qq_chi2(ax, d2_test, d, "normal data", "C0")
qq_chi2(ax, d2t, d, r"$t_3$ data", "C3")
ax.set_ylim(0, 60); ax.legend(); ax.set_title(r"Stress test: heavy tails break $d^2\sim\chi^2_5$")
fig.tight_layout(); fig.savefig(FIG / "sim_stress_t3_qq.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------------------
# Part E2: stress test - an exact copy of a feature (Assumption 3 broken, Q5)
# ---------------------------------------------------------------------------
header("PART E2: stress test - duplicated feature makes Sigma_hat singular (Q5)")
X_dup = add_duplicate_feature(X, j=0)
_, S_dup = my_mle(X_dup)
lam_dup, _ = my_jacobi_eig(S_dup)
print("eigenvalues of Sigma_hat (Jacobi):", lam_dup)
print(f"cond(Sigma_hat) = {np.linalg.cond(S_dup):.3g} (numpy.linalg.cond, check only)")
try:
    my_ldl(S_dup)
    ldl_stopped = False
    print("LDL^T succeeded (unexpected)")
except ValueError as e:
    ldl_stopped = True
    print("my_ldl stops:", e)
eps = 1e-3 * np.trace(S_dup) / S_dup.shape[0]             # ridge: Sigma_hat + eps I
S_ridge = S_dup + eps * np.eye(S_dup.shape[0])
L_r = my_cholesky(S_ridge)
d2_r = my_mahalanobis2(X_dup, mu_hat=X_dup.mean(axis=0), L=L_r)
print(f"fix 1, ridge eps = {eps:.4f}: cond = {np.linalg.cond(S_ridge):.3g}; "
      f"mean training d^2 = {d2_r.mean():.3f} (d = 6 without ridge)")
print("fix 2, drop the copy: back to the d = 5 model above "
      f"(cond = {np.linalg.cond(S_hat):.3g})")
# almost collinear: copy plus tiny noise
rng = np.random.default_rng(1)
X_near = np.column_stack([X, X[:, 0] + 1e-4 * rng.normal(size=n)])
_, S_near = my_mle(X_near)
L_near = my_cholesky(S_near)
d2_near = my_mahalanobis2(X_near, X_near.mean(axis=0), L_near)
d2_near_lib = np.sum((X_near - X_near.mean(0)) @ np.linalg.inv(S_near) * (X_near - X_near.mean(0)), 1)
print(f"almost collinear (noise 1e-4): cond = {np.linalg.cond(S_near):.3g}")
a_near, r_near = report("  d^2 (Cholesky) vs numpy.inv", d2_near, d2_near_lib)
table_row("Simulation (stress: copy)", "LDL pivot D_6", "0 (singular)",
          f"cond = {np.linalg.cond(S_dup):.0e}, LDL stops", ldl_stopped)
table_row("Simulation (stress: near copy)", "d^2", "numpy.linalg.inv",
          f"rel = {r_near:.0e} (cond {np.linalg.cond(S_near):.0e})", r_near < 1e-6)

# ---------------------------------------------------------------------------
# Part E3: stress test - small sample n = 200, d = 21 (Assumption 5)
# ---------------------------------------------------------------------------
header("PART E3: small sample (n = 200, d = 21), held-out points, 200 repetitions")
dD, nD, reps, m_new = 21, 200, 200, 2000
alphas = (0.10, 0.05, 0.01)
tau_chi = [my_threshold(a, dD) for a in alphas]
tau_F = [my_threshold_F(a, dD, nD) for a in alphas]
for a, t1, t2 in zip(alphas, tau_chi, tau_F):
    tF_lib = dD * (nD + 1) / (nD - dD) * stats.f.ppf(1 - a, dD, nD - dD)
    print(f"alpha = {a}: tau_chi2 = {t1:.3f} (scipy {stats.chi2.ppf(1 - a, dD):.3f}),  "
          f"tau_F = {t2:.3f} (scipy {tF_lib:.3f})")
far_c = np.zeros((reps, 3)); far_f = np.zeros((reps, 3))
for r in range(reps):
    XD, _, _ = simulate_normal(d=dD, n=nD + m_new, seed=seed, rep=r)
    m_r, S_r = my_mle(XD[:nD])
    d2_new = my_mahalanobis2(XD[nD:], m_r, my_cholesky(S_r))
    far_c[r] = [np.mean(d2_new > t) for t in tau_chi]
    far_f[r] = [np.mean(d2_new > t) for t in tau_F]
print(f"{'alpha':>7}{'chi^2 threshold':>17}{'F threshold':>14}")
for k, a in enumerate(alphas):
    print(f"{a:>7.2f}{far_c[:, k].mean():>17.4f}{far_f[:, k].mean():>14.4f}")
table_row("Simulation (stress: n=200, d=21)", "false-alarm rate", "alpha = 0.01",
          f"chi2 {far_c[:, 2].mean():.3f}, F {far_f[:, 2].mean():.3f}",
          abs(far_f[:, 2].mean() - 0.01) < 0.002)

# ---------------------------------------------------------------------------
# Part F: assumption checks on simulated data (hold by construction)
# ---------------------------------------------------------------------------
res_big = assumption_checks(X, X_test, "simulation d = 5, n = 5000")
XD, _, _ = simulate_normal(d=dD, n=nD + nD, seed=seed, rep=0)
res_small = assumption_checks(XD[:nD], XD[nD:], "simulation d = 21, n = 200 (real-data sizes)")
qq_normal_grid(XD[:nD], [f"x{j + 1}" for j in range(dD)], "sim_feature_qq.png",
               "Simulation (n = 200, d = 21): normal Q-Q plot of each feature")

print_table()
print(f"\nFigures saved to {FIG}")
