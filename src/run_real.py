"""
DS5020 Final Project - Iteration 3: real data and library comparison
Project 9: Anomaly Detection with the Multivariate Normal Distribution
CMU Keystroke Dynamics Benchmark (Killourhy and Maxion, 2009)

Protocol (Iteration 3 handout, Project 9 box):
  * genuine user u: fit on the first 200 repetitions (sessions 1-4), test on the
    remaining 200 (sessions 5-8);
  * impostors: the first 50 repetitions (session 1) of every other subject, 50 x 50 = 2500;
  * false-alarm rate (genuine test typings flagged) and detection rate (impostor
    typings flagged) at alpha = 0.01 and 0.05, with and without the log transform.
Features: the d = 21 hold (H) and keydown-keydown (DD) times; the 10 UD columns are
dropped because UD = DD - H (Iteration 2, Assumption 3).

  Part A  cleaning steps
  Part B  one user (s002) step by step, every from-scratch result vs the library
  Part C  all 51 users: library differences, rates for 3 transforms x 2 thresholds
  Part D  assumption checks on the real data (Iteration 3 Q3, Q4, Q5)

Run from the project root:  python3 src/run_real.py
(needs data/DSL-StrongPasswordData.csv, see README)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from load import load_raw, feature_columns, split_user
from methods import (my_log_transform, my_mle, my_cholesky, my_mahalanobis2, my_jacobi_eig,
                     my_inv_sqrt, my_whiten_distance2, my_inverse, my_threshold, my_threshold_F,
                     my_rates)
from diagnostics import (report, table_row, print_table, header, qq_chi2, qq_normal_grid,
                         assumption_checks, check_pd, FIG)

np.set_printoptions(precision=4, suppress=True)
USER = "s002"
ALPHAS = (0.01, 0.05)
N_IMP_REPS = 50
H, DD, UD = feature_columns()
FEATS = H + DD
d = len(FEATS)


def features(frame, mode):
    """'raw': seconds;  'logH': log of the 11 hold times only;  'log': log of all 21."""
    Xr = frame[FEATS].to_numpy(dtype=float)
    if mode == "raw":
        return Xr
    if mode == "logH":
        X = Xr.copy()
        X[:, :len(H)] = my_log_transform(Xr[:, :len(H)])
        return X
    return my_log_transform(Xr)


MODES = {"raw": "raw seconds", "logH": "log hold times only", "log": "log all 21"}

# ---------------------------------------------------------------------------
# Part A: cleaning
# ---------------------------------------------------------------------------
header("PART A: cleaning steps")
df = load_raw()
print(f"rows x columns: {df.shape[0]} x {df.shape[1]}, subjects: {df['subject'].nunique()}")
print(f"missing values: {int(df.isna().sum().sum())}  -> 0 rows removed")
print(f"duplicate rows: {int(df.duplicated().sum())}  -> 0 rows removed")
print(f"dropped columns: the {len(UD)} UD latencies (UD = DD - H, exact linear dependence)")
print(f"kept d = {d} features: {len(H)} H + {len(DD)} DD; all > 0: "
      f"{bool((df[FEATS] > 0).all().all())} (so the log is defined, no rows removed)")
train, test_g, test_i = split_user(df, USER, n_imp_reps=N_IMP_REPS)
print(f"split for {USER}: train = sessions 1-4 ({len(train)}), genuine test = sessions 5-8 "
      f"({len(test_g)}), impostors = reps 1-{N_IMP_REPS} of session 1 of "
      f"{test_i['subject'].nunique()} other subjects ({len(test_i)})")

# ---------------------------------------------------------------------------
# Part B: one user, every step vs the library
# ---------------------------------------------------------------------------
header(f"PART B: genuine user {USER}, log features, from-scratch vs library")
X_tr, X_g, X_i = features(train, "log"), features(test_g, "log"), features(test_i, "log")
n = len(X_tr)
mu_hat, S_hat = my_mle(X_tr)
L = my_cholesky(S_hat)
lam, Q = my_jacobi_eig(S_hat)
W = my_inv_sqrt(lam, Q)
X_all = np.vstack([X_tr, X_g, X_i])
d2 = my_mahalanobis2(X_all, mu_hat, L)
d2_W = my_whiten_distance2(X_all, mu_hat, W)
diff = X_all - mu_hat
d2_lib = np.sum(diff @ np.linalg.inv(S_hat) * diff, axis=1)
lam_lib, Q_lib = np.linalg.eigh(S_hat)
sgn = np.sign(np.sum(Q * Q_lib, axis=0))
print(f"mu_hat (first 5) = {mu_hat[:5]}")
print(f"Sigma_hat: eigenvalues {lam.min():.4g} .. {lam.max():.4g}, cond = {lam.max() / lam.min():.1f}")
comparisons = [
    ("Sigma_hat vs np.cov(bias=True)", S_hat, np.cov(X_tr.T, bias=True)),
    ("Cholesky vs numpy", L, np.linalg.cholesky(S_hat)),
    ("inverse (LDL) vs numpy.inv", my_inverse(S_hat), np.linalg.inv(S_hat)),
    ("eigenvalues (Jacobi) vs eigh", lam, lam_lib),
    ("eigenvectors vs eigh (signs aligned)", Q * sgn, Q_lib),
    ("d^2 (Cholesky) vs numpy.inv", d2, d2_lib),
    ("d^2 (W) vs numpy.inv", d2_W, d2_lib),
]
for name, mine, ref in comparisons:
    a, r = report(name, mine, ref)
    if not name.startswith("Sigma_hat"):
        table_row(f"Real data ({USER})", name.split(" vs ")[0], name.split(" vs ")[-1],
                  f"max = {a:.0e}, rel = {r:.0e}", r < 1e-8)
for a in ALPHAS:
    t_m, t_l = my_threshold(a, d), stats.chi2.ppf(1 - a, d)
    f_m = my_threshold_F(a, d, n)
    f_l = d * (n + 1) / (n - d) * stats.f.ppf(1 - a, d, n - d)
    print(f"alpha = {a}: tau_chi2 ours {t_m:.6f} scipy {t_l:.6f} | tau_F ours {f_m:.6f} scipy {f_l:.6f}")
table_row(f"Real data ({USER})", "tau_F (alpha 0.01)", "scipy.stats.f",
          f"{abs(my_threshold_F(0.01, d, n) - d * (n + 1) / (n - d) * stats.f.ppf(0.99, d, n - d)):.0e}",
          True)

print(f"\nRates for {USER} (false alarm = genuine sessions 5-8 flagged; detection = impostors flagged)")
print(f"{'features':<22}{'threshold':<10}{'alpha':>6}{'false alarm':>13}{'detection':>11}")
for mode, label in MODES.items():
    A, G, I = features(train, mode), features(test_g, mode), features(test_i, mode)
    m_, S_ = my_mle(A)
    L_ = my_cholesky(S_)
    dg, di = my_mahalanobis2(G, m_, L_), my_mahalanobis2(I, m_, L_)
    for thr in ("chi2", "F"):
        for a in ALPHAS:
            tau = my_threshold(a, d) if thr == "chi2" else my_threshold_F(a, d, len(A))
            far, det = my_rates(dg, di, tau)
            print(f"{label:<22}{thr:<10}{a:>6.2f}{far:>13.3f}{det:>11.3f}")

# ---------------------------------------------------------------------------
# Part C: all 51 users
# ---------------------------------------------------------------------------
users = sorted(df["subject"].unique())
header(f"PART C: all {len(users)} genuine users")
rows = []                 # (mode, thr, alpha, user, far, det)
max_rel_lib = 0.0
conds = {m: [] for m in MODES}
for u in users:
    tr, tg, ti = split_user(df, u, n_imp_reps=N_IMP_REPS)
    for mode in MODES:
        A, G, I = features(tr, mode), features(tg, mode), features(ti, mode)
        m_, S_ = my_mle(A)
        L_ = my_cholesky(S_)
        dg, di = my_mahalanobis2(G, m_, L_), my_mahalanobis2(I, m_, L_)
        conds[mode].append(check_pd(S_)[2])
        if mode == "log":                                   # library check for every user
            Vg = G - m_
            ref = np.sum(Vg @ np.linalg.inv(S_) * Vg, axis=1)
            max_rel_lib = max(max_rel_lib, np.linalg.norm(dg - ref) / np.linalg.norm(ref))
        for thr in ("chi2", "F"):
            for a in ALPHAS:
                tau = my_threshold(a, d) if thr == "chi2" else my_threshold_F(a, d, len(A))
                rows.append((mode, thr, a, u) + my_rates(dg, di, tau))
print(f"max relative difference of d^2 vs numpy.inv over all {len(users)} users: {max_rel_lib:.1e}")
table_row(f"Real data (all {len(users)} users)", "d^2 (Cholesky)", "numpy.inv",
          f"max rel = {max_rel_lib:.0e}", max_rel_lib < 1e-8)
print("\ncond(Sigma_hat) over users:")
for mode, label in MODES.items():
    c = np.array(conds[mode])
    print(f"  {label:<22} median {np.median(c):9.1f}   max {c.max():9.1f}")
print(f"\nMean (and median) over {len(users)} users:")
print(f"{'features':<22}{'threshold':<10}{'alpha':>6}{'false alarm':>18}{'detection':>18}")
summary = {}
for mode, label in MODES.items():
    for thr in ("chi2", "F"):
        for a in ALPHAS:
            sel = [r for r in rows if r[0] == mode and r[1] == thr and r[2] == a]
            far = np.array([r[4] for r in sel]); det = np.array([r[5] for r in sel])
            summary[(mode, thr, a)] = (far.mean(), det.mean())
            print(f"{label:<22}{thr:<10}{a:>6.2f}{far.mean():>10.3f} ({np.median(far):.3f})"
                  f"{det.mean():>10.3f} ({np.median(det):.3f})")

# ---------------------------------------------------------------------------
# Part D: assumption checks on the real data (simulated first: check_simulation.py, Part F)
# ---------------------------------------------------------------------------
res = assumption_checks(X_tr, X_g, f"real data, {USER}, log features", names=FEATS)
d2_tr = res["d2_train"]; d2_g = res["d2_test"]
d2_i = my_mahalanobis2(X_i, mu_hat, L)

# A1 and A2 over all users: drift (held-out mean d^2, Hotelling test between the two
# halves of training) and the number of features that fail a normality test
mean_te, p_drift, n_bad = [], [], {"raw": [], "log": []}
for u in users:
    tr, tg, _ = split_user(df, u, n_imp_reps=1)
    A = features(tr, "log")
    m_, S_ = my_mle(A)
    mean_te.append(my_mahalanobis2(features(tg, "log"), m_, my_cholesky(S_)).mean())
    h = len(A) // 2
    m1, S1 = my_mle(A[:h]); m2, S2 = my_mle(A[h:])
    S_p = (h * S1 + (len(A) - h) * S2) / (len(A) - 2)
    dm = m1 - m2
    T2 = dm @ np.linalg.solve(S_p * (1 / h + 1 / (len(A) - h)), dm)
    p_drift.append(stats.f.sf((len(A) - d - 1) / (d * (len(A) - 2)) * T2, d, len(A) - d - 1))
    for mode in ("raw", "log"):
        Xm = features(tr, mode)
        n_bad[mode].append(sum(stats.normaltest(Xm[:, j]).pvalue < 0.01 for j in range(d)))
mean_te, p_drift = np.array(mean_te), np.array(p_drift)
print(f"\n[A1] all users: mean held-out d^2 median {np.median(mean_te):.1f} "
      f"(iid value {d * (n + 1) / (n - d - 2):.1f}); users above twice that: "
      f"{int(np.sum(mean_te > 2 * d * (n + 1) / (n - d - 2)))} of {len(users)}")
print(f"[A1] all users: Hotelling test sessions 1-2 vs 3-4 rejects at 1% for "
      f"{int(np.sum(p_drift < 0.01))} of {len(users)} users")
for mode in ("raw", "log"):
    b = np.array(n_bad[mode])
    print(f"[A2] all users, {MODES[mode]}: features rejecting normality at 1%: "
          f"median {np.median(b):.0f} of {d} (min {b.min()}, max {b.max()})")

# A2 per feature, raw vs log (Iteration 3, Q4)
for mode in ("raw", "log"):
    Xm = features(train, mode)
    pv = np.array([stats.normaltest(Xm[:, j]).pvalue for j in range(d)])
    sk = stats.skew(Xm, axis=0)
    worst = np.argsort(pv)[:4]
    print(f"[A2] {USER} {MODES[mode]}: {int(np.sum(pv < 0.01))} of {d} features reject normality "
          f"at 1%; median |skew| {np.median(np.abs(sk)):.2f}; worst: "
          + ", ".join(f"{FEATS[j]} (skew {sk[j]:.2f})" for j in worst))
qq_normal_grid(features(train, "raw"), FEATS, "real_feature_qq_raw.png",
               f"{USER}, training typings, raw seconds: normal Q-Q plot of each feature")
qq_normal_grid(X_tr, FEATS, "real_feature_qq_log.png",
               f"{USER}, training typings, log seconds: normal Q-Q plot of each feature")

# A3 / Q5: condition number, raw vs log, ridge
eps = 1e-3 * np.trace(S_hat) / d
S_r = S_hat + eps * np.eye(d)
L_r = my_cholesky(S_r)
far_r, det_r = my_rates(my_mahalanobis2(X_g, mu_hat, L_r), my_mahalanobis2(X_i, mu_hat, L_r),
                        my_threshold(0.01, d))
far_0, det_0 = my_rates(d2_g, d2_i, my_threshold(0.01, d))
print(f"[A3] {USER}: cond(Sigma_hat) log {check_pd(S_hat)[2]:.1f}, ridge eps = {eps:.2e} -> "
      f"{check_pd(S_r)[2]:.1f}; alpha 0.01 rates (FA, DR) {far_0:.3f}, {det_0:.3f} -> "
      f"{far_r:.3f}, {det_r:.3f}")

# A4: refit without the 5 most extreme training typings
keep = np.argsort(d2_tr)[:-5]
m4, S4 = my_mle(X_tr[keep]); L4 = my_cholesky(S4)
far4, det4 = my_rates(my_mahalanobis2(X_g, m4, L4), my_mahalanobis2(X_i, m4, L4),
                      my_threshold_F(0.01, d, len(keep)))
far0F, det0F = my_rates(d2_g, d2_i, my_threshold_F(0.01, d, n))
print(f"[A4] {USER}: drop 5 largest training d^2 -> alpha 0.01 (F) rates (FA, DR) "
      f"{far0F:.3f}, {det0F:.3f} -> {far4:.3f}, {det4:.3f}")

# Q3 figure: d^2 vs chi^2_21
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
qq_chi2(axes[0], d2_tr, d, "training (sessions 1-4)", "C0")
qq_chi2(axes[0], d2_g, d, "genuine test (sessions 5-8)", "C2")
axes[0].set_title(rf"{USER}: $d^2$ vs $\chi^2_{{{d}}}$ (genuine typings)"); axes[0].legend()
bins = np.linspace(0, np.percentile(d2_i, 95), 60)
axes[1].hist(d2_g, bins=bins, density=True, alpha=0.6, label="genuine test")
axes[1].hist(d2_i, bins=bins, density=True, alpha=0.6, label="impostors")
g = np.linspace(0.1, bins[-1], 300)
axes[1].plot(g, stats.chi2.pdf(g, d), "k-", lw=1.5, label=rf"$\chi^2_{{{d}}}$")
for a, ls in zip(ALPHAS, ("--", ":")):
    axes[1].axvline(my_threshold(a, d), color="r", ls=ls, label=rf"$\tau$, $\alpha$={a}")
axes[1].set_xlabel(r"$d^2$"); axes[1].legend(fontsize=8)
axes[1].set_title(f"{USER}: genuine vs impostor distances")
fig.tight_layout(); fig.savefig(FIG / "real_d2_qq.png", dpi=150); plt.close(fig)

# rates figure
fig, ax = plt.subplots(figsize=(7, 4))
labels, fa, de = [], [], []
for mode in MODES:
    for thr in ("chi2", "F"):
        labels.append(f"{mode}\n{thr}")
        fa.append(summary[(mode, thr, 0.01)][0]); de.append(summary[(mode, thr, 0.01)][1])
xpos = np.arange(len(labels))
ax.bar(xpos - 0.2, fa, 0.4, label="false-alarm rate")
ax.bar(xpos + 0.2, de, 0.4, label="detection rate")
ax.axhline(0.01, color="k", lw=0.8, ls="--")
ax.set_xticks(xpos); ax.set_xticklabels(labels)
ax.set_title(r"All 51 users, $\alpha = 0.01$ (mean over users)"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "real_rates.png", dpi=150); plt.close(fig)

for mode in ("raw", "log"):
    for thr in ("chi2", "F"):
        fa_, de_ = summary[(mode, thr, 0.01)]
        table_row(f"Real data (all users, {mode})", f"false-alarm rate ({thr})", "alpha = 0.01",
                  f"{fa_:.3f} (detection {de_:.3f})", abs(fa_ - 0.01) <= 0.01)
print("\nThe real-data false-alarm rates are NOT expected to pass: the checks in Part D show")
print("drift between sessions (A1) and non-normal features (A2), which the simulation does not have.")

print_table()
print(f"\nFigures saved to {FIG}")
