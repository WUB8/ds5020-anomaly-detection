"""
DS5020 Final Project - Iteration 3: verification and assumption-check helpers
These are checks, not the method, so they may use numpy.linalg and SciPy.
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

# Rows of the verification table, collected by every script that calls report()/table_row()
TABLE = []


def report(name, mine, ref):
    """Print max |mine - ref| and the relative error ||mine - ref|| / ||ref||."""
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    abs_err = np.abs(mine - ref).max()
    nref = np.linalg.norm(ref)
    rel_err = np.linalg.norm(mine - ref) / nref if nref > 0 else np.linalg.norm(mine - ref)
    print(f"{name:32s} max|diff| = {abs_err:.2e}   rel = {rel_err:.2e}")
    return abs_err, rel_err


def table_row(data, quantity, compared_with, difference, passed):
    TABLE.append((data, quantity, compared_with, difference, passed))


def print_table(title="VERIFICATION TABLE ROWS"):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
    w = [max(len(str(r[i])) for r in TABLE + [("Data", "Quantity", "Compared with", "Difference", "")])
         for i in range(4)]
    print(f"{'Data':{w[0]}}  {'Quantity':{w[1]}}  {'Compared with':{w[2]}}  {'Difference':{w[3]}}  Pass?")
    for r in TABLE:
        print(f"{r[0]:{w[0]}}  {r[1]:{w[1]}}  {r[2]:{w[2]}}  {r[3]:{w[3]}}  {'yes' if r[4] else 'NO'}")


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Assumption checks (Iteration 2, Assumptions 1-5), run on simulated data first
# ---------------------------------------------------------------------------
def check_normal_features(X, names=None):
    """Assumption 2 (marginal part): skewness, excess kurtosis and the
    D'Agostino-Pearson normality test p-value for each feature."""
    n, d = X.shape
    names = names or [f"x{j + 1}" for j in range(d)]
    skew = stats.skew(X, axis=0)
    kurt = stats.kurtosis(X, axis=0)
    pval = np.array([stats.normaltest(X[:, j]).pvalue for j in range(d)])
    return dict(names=names, skew=skew, kurt=kurt, pval=pval)


def check_pd(S):
    """Assumption 3: eigenvalues and condition number of Sigma_hat (check only)."""
    lam = np.linalg.eigvalsh(S)
    cond = lam.max() / lam.min() if lam.min() > 0 else np.inf
    return lam.min(), lam.max(), cond


def print_normal_summary(res, label, alpha=0.01):
    bad = res["pval"] < alpha
    print(f"[A2] {label}: features with normality p < {alpha}: {int(bad.sum())} of {len(bad)}; "
          f"|skew| median {np.median(np.abs(res['skew'])):.2f} (max {np.abs(res['skew']).max():.2f}); "
          f"excess kurtosis median {np.median(res['kurt']):.2f} (max {res['kurt'].max():.2f})")
    return bad


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def qq_chi2(ax, d2, d, label, color="C0"):
    """Q-Q plot of sorted d^2 against chi^2_d quantiles (Iteration 3, Q3)."""
    m = len(d2)
    probs = (np.arange(1, m + 1) - 0.5) / m
    q = stats.chi2.ppf(probs, df=d)
    ax.plot(q, np.sort(d2), ".", ms=3, color=color, label=label)
    lim = q[-1]
    ax.plot([0, lim], [0, lim], "k-", lw=1)
    ax.set_xlabel(rf"$\chi^2_{{{d}}}$ quantiles")
    ax.set_ylabel(r"sorted $d^2$")


def qq_normal_grid(X, names, fname, title):
    """Normal Q-Q plot for each feature (Iteration 3, Q4)."""
    d = X.shape[1]
    ncol = 7 if d > 8 else d
    nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.1 * nrow), squeeze=False)
    for j, ax in enumerate(axes.ravel()):
        if j >= d:
            ax.axis("off")
            continue
        x = np.sort((X[:, j] - X[:, j].mean()) / X[:, j].std())
        q = stats.norm.ppf((np.arange(1, len(x) + 1) - 0.5) / len(x))
        ax.plot(q, x, ".", ms=2)
        ax.plot([-3.5, 3.5], [-3.5, 3.5], "r-", lw=0.8)
        ax.set_title(names[j], fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def assumption_checks(X_train, X_test, label, names=None, alpha=0.01):
    """Run the checks for Assumptions 1-5 of the report on one training set and one
    held-out set of NORMAL (genuine) data. Uses our methods for d^2 and SciPy only
    for reference distributions."""
    from methods import my_mle, my_cholesky, my_mahalanobis2, my_ldl
    n, d = X_train.shape
    out = {}
    header(f"ASSUMPTION CHECKS: {label}  (n = {n}, d = {d}, held-out m = {len(X_test)})")
    mu_hat, S_hat = my_mle(X_train)
    L = my_cholesky(S_hat)
    d2_tr = my_mahalanobis2(X_train, mu_hat, L)
    d2_te = my_mahalanobis2(X_test, mu_hat, L)

    # A1: iid / no drift -- two-sample Hotelling T^2, first vs second half of training:
    #   T^2 = (m1 - m2)^T [S_p (1/n1 + 1/n2)]^{-1} (m1 - m2),  S_p pooled unbiased covariance,
    #   (n1 + n2 - d - 1) / (d (n1 + n2 - 2)) T^2 ~ F_{d, n1 + n2 - d - 1}
    h = n // 2
    A, B = X_train[:h], X_train[h:]
    n1, n2 = len(A), len(B)
    m1, S1 = my_mle(A); m2, S2 = my_mle(B)
    S_p = (n1 * S1 + n2 * S2) / (n1 + n2 - 2)
    diff = m1 - m2
    T2 = float(diff @ np.linalg.solve(S_p * (1 / n1 + 1 / n2), diff))
    F_shift = (n1 + n2 - d - 1) / (d * (n1 + n2 - 2)) * T2
    p_T = stats.f.sf(F_shift, d, n1 + n2 - d - 1)
    exp_te = d * (n + 1) / (n - d - 2)                  # E[d^2] for a new point (F result)
    print(f"[A1] mean shift, 1st vs 2nd half of training: Hotelling T^2 = {T2:.1f}, p = {p_T:.2g}")
    print(f"[A1] mean d^2: training {d2_tr.mean():.2f} (= d = {d} by Prop.), held-out "
          f"{d2_te.mean():.2f} (expected {exp_te:.2f} for iid normal data)")
    out.update(T_shift=T2, p_shift=p_T, mean_d2_test=d2_te.mean(), exp_d2_test=exp_te)

    # A2: normality -- marginals and d^2 vs chi^2_d
    res = check_normal_features(X_train, names)
    bad = print_normal_summary(res, "marginal normality", alpha)
    ks = stats.kstest(d2_te, "chi2", args=(d,))
    c = d * (n + 1) / (n - d)                           # held-out d^2 / c ~ F_{d, n-d}
    ks_F = stats.kstest(d2_te / c, "f", args=(d, n - d))
    print(f"[A2] KS test of held-out d^2: vs chi^2_{d} p = {ks.pvalue:.2g};  "
          f"vs exact small-sample law (d^2/{c:.3f} ~ F_({d},{n - d})) p = {ks_F.pvalue:.2g}")
    out.update(normal=res, n_bad=int(bad.sum()), ks_p=ks.pvalue, ks_F_p=ks_F.pvalue,
               d2_train=d2_tr, d2_test=d2_te)

    # A3: positive definite
    lmin, lmax, cond = check_pd(S_hat)
    _, D = my_ldl(S_hat)
    print(f"[A3] lambda_min = {lmin:.3g}, lambda_max = {lmax:.3g}, cond(Sigma_hat) = {cond:.3g}; "
          f"smallest LDL pivot = {D.min():.3g}")
    out.update(lmin=lmin, cond=cond)

    # A4: clean training data -- drop the 5 largest training d^2 and refit
    keep = np.argsort(d2_tr)[:-5]
    mu2, S2 = my_mle(X_train[keep])
    dS = np.linalg.norm(S2 - S_hat) / np.linalg.norm(S_hat)
    print(f"[A4] largest training d^2: {np.sort(d2_tr)[-5:].round(1)} "
          f"(chi^2_{d} 99.9% point = {stats.chi2.ppf(0.999, d):.1f}); "
          f"refit without them changes Sigma_hat by {100 * dS:.1f}% (Frobenius)")
    out.update(top_d2=np.sort(d2_tr)[-5:], dS_clean=dS)

    # A5: large sample
    print(f"[A5] n / d = {n / d:.1f};  parameters in Sigma_hat = d(d+1)/2 = {d * (d + 1) // 2}")
    out.update(ratio=n / d)
    return out
