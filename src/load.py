"""
DS5020 Final Project - Iteration 2: load and describe the real dataset
CMU Keystroke Dynamics Benchmark (Killourhy and Maxion, 2009)

Download the data first (see README):
    curl -L -o data/DSL-StrongPasswordData.csv \
        https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv

Run from the project root:  python3 src/load.py
Figures are saved to figures/.

The functions load_raw, feature_columns, feature_matrix and split_user are
reused in Iteration 3.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "DSL-StrongPasswordData.csv"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

KEYS = ["period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l", "Return"]


# ---------------------------------------------------------------------------
# Reusable helpers
# ---------------------------------------------------------------------------
def load_raw(path=DATA):
    """Read the CSV. One row = one typing of '.tie5Roanl' + Return."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - see README for the download command")
    return pd.read_csv(path)


def feature_columns():
    """Column names of H (11), DD (10) and UD (10) features, in key order."""
    H = [f"H.{k}" for k in KEYS]
    DD = [f"DD.{a}.{b}" for a, b in zip(KEYS[:-1], KEYS[1:])]
    UD = [f"UD.{a}.{b}" for a, b in zip(KEYS[:-1], KEYS[1:])]
    return H, DD, UD


def feature_matrix(df, log=True):
    """The d = 21 model features: 11 hold times + 10 DD latencies (UD dropped).
    Returns an (n x 21) numpy array, log-transformed by default."""
    H, DD, _ = feature_columns()
    X = df[H + DD].to_numpy(dtype=float)
    return np.log(X) if log else X


def split_user(df, user, n_imp_reps=5):
    """Benchmark protocol for one genuine user:
    train  = user's sessions 1-4 (200 typings)
    test_g = user's sessions 5-8 (200 typings, normal)
    test_i = first n_imp_reps typings of session 1 of every other subject (impostors)."""
    me = df["subject"] == user
    train = df[me & (df["sessionIndex"] <= 4)]
    test_g = df[me & (df["sessionIndex"] >= 5)]
    test_i = df[~me & (df["sessionIndex"] == 1) & (df["rep"] <= n_imp_reps)]
    return train, test_g, test_i


# ---------------------------------------------------------------------------
# Describe the data
# ---------------------------------------------------------------------------
def main():
    df = load_raw()
    H, DD, UD = feature_columns()
    timing = H + DD + UD

    print("=" * 70 + "\nSIZE AND TYPES\n" + "=" * 70)
    print(f"rows x columns: {df.shape[0]} x {df.shape[1]}")
    print(f"timing features: {len(H)} H + {len(DD)} DD + {len(UD)} UD = {len(timing)}")
    print("dtypes:", df.dtypes.value_counts().to_dict())
    print(f"subjects: {df['subject'].nunique()}  "
          f"(e.g. {', '.join(sorted(df['subject'].unique())[:3])}, ...)")
    print(f"sessionIndex: {df['sessionIndex'].min()}-{df['sessionIndex'].max()}, "
          f"rep: {df['rep'].min()}-{df['rep'].max()}")
    per_subj = df.groupby("subject").size()
    print(f"rows per subject: min {per_subj.min()}, max {per_subj.max()}")
    assert set(timing) | {"subject", "sessionIndex", "rep"} == set(df.columns)

    print("\n" + "=" * 70 + "\nMISSING VALUES AND RANGES\n" + "=" * 70)
    print(f"missing values (total): {int(df.isna().sum().sum())}")
    print(f"duplicate rows: {int(df.duplicated().sum())}")
    print(f"min H  = {df[H].min().min():.4f} s,  min DD = {df[DD].min().min():.4f} s "
          f"(all > 0 so log is defined: {bool((df[H + DD] > 0).all().all())})")
    neg = (df[UD] < 0).to_numpy()
    print(f"negative UD values: {int(neg.sum())} of {neg.size} "
          f"({100 * neg.mean():.2f}%), in {int(neg.any(axis=1).sum())} rows")
    print(f"max timing = {df[timing].max().max():.4f} s")

    # Linear dependence UD = DD - H (Assumption 3)
    diffs = [np.abs(df[f"UD.{a}.{b}"] - (df[f"DD.{a}.{b}"] - df[f"H.{a}"])).max()
             for a, b in zip(KEYS[:-1], KEYS[1:])]
    print(f"\n[A3] max |UD - (DD - H)| over all rows and key pairs: {max(diffs):.2e} s "
          "(zero up to floating-point rounding)")

    print("\n" + "=" * 70 + "\nSUMMARY STATISTICS (all subjects)\n" + "=" * 70)
    show = ["H.t", "DD.t.i", "UD.t.i", "H.Return", "DD.five.Shift.r"]
    rows = []
    for c in show:
        x = df[c].to_numpy()
        r = {"feature": c, "mean": x.mean(), "std": x.std(ddof=1), "min": x.min(),
             "max": x.max(), "skew": stats.skew(x)}
        r["skew(log)"] = stats.skew(np.log(x)) if (x > 0).all() else np.nan
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    sk_raw = stats.skew(df[H + DD].to_numpy(), axis=0)
    sk_log = stats.skew(np.log(df[H + DD].to_numpy()), axis=0)
    print(f"\nmedian skewness over the 21 model features: raw {np.median(sk_raw):.2f}, "
          f"log {np.median(sk_log):.2f}")

    # Covariance rank with 31 vs 21 features (Assumption 3), for one user
    user = "s002"
    train, test_g, test_i = split_user(df, user)
    print("\n" + "=" * 70 + f"\nONE GENUINE USER: {user}\n" + "=" * 70)
    print(f"train: {len(train)},  genuine test: {len(test_g)},  impostor test: {len(test_i)} "
          f"({test_i['subject'].nunique()} other subjects)")
    for name, cols in [("31 raw features", timing), ("21 model features", H + DD)]:
        Xu = train[cols].to_numpy(dtype=float)
        Xc = Xu - Xu.mean(axis=0)
        lam = np.linalg.eigvalsh(Xc.T @ Xc / len(Xu))
        n_zero = int(np.sum(lam < 1e-6 * lam.max()))
        print(f"[A3] {name}: n = {len(Xu)}, d = {len(cols)}, "
              f"near-zero eigenvalues of Sigma_hat = {n_zero}, "
              f"condition number = "
              f"{'infinite (singular)' if n_zero else f'{lam.max() / lam.min():.3g}'}")
    Xlog = feature_matrix(train)
    lam = np.linalg.eigvalsh(np.cov(Xlog.T, bias=True))
    print(f"[A3] 21 log features: smallest eigenvalue {lam.min():.3g}, "
          f"condition number {lam.max() / lam.min():.3g}")

    # ---------------------------------------------------------------------
    # Plot 1: raw vs log hold time (Assumption 2: normality)
    # ---------------------------------------------------------------------
    feat = "H.t"
    x = df.loc[df["subject"] == user, feat].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, vals, lab in [(axes[0], x, "seconds"), (axes[1], np.log(x), "log seconds")]:
        ax.hist(vals, bins=40, density=True, alpha=0.6)
        g = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(g, stats.norm.pdf(g, vals.mean(), vals.std(ddof=1)), "k-", lw=2,
                label="normal fit")
        ax.set_xlabel(f"{feat} ({lab})")
        ax.set_title(f"{'raw' if lab == 'seconds' else 'log'}: skewness = {stats.skew(vals):.2f}")
        ax.legend()
    axes[0].set_ylabel("density")
    fig.suptitle(f"Hold time of key 't', subject {user} (400 typings)")
    fig.tight_layout(); fig.savefig(FIG / "real_hold_time_hist.png", dpi=150); plt.close(fig)

    # ---------------------------------------------------------------------
    # Plot 2: total typing time vs repetition (Assumption 1: drift / practice)
    # ---------------------------------------------------------------------
    df["total_DD"] = df[DD].sum(axis=1)
    df["rep_all"] = (df["sessionIndex"] - 1) * 50 + df["rep"]
    subjects = sorted(df["subject"].unique())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    cmap = plt.get_cmap("viridis", 8)
    for ax, s in zip(axes.ravel(), subjects):
        d_s = df[df["subject"] == s]
        ax.scatter(d_s["rep_all"], d_s["total_DD"], c=d_s["sessionIndex"], cmap=cmap,
                   vmin=0.5, vmax=8.5, s=6)
        sess_mean = d_s.groupby("sessionIndex")["total_DD"].median()
        ax.plot((sess_mean.index - 0.5) * 50, sess_mean.values, "r-o", ms=3,
                label="session median")
        ax.set_yscale("log"); ax.set_title(s); ax.legend(loc="upper right", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("repetition (1-400, sessions 1-8)")
    for ax in axes[:, 0]:
        ax.set_ylabel("total DD time (s, log scale)")
    fig.suptitle("Typing gets faster across sessions (practice effect)")
    fig.tight_layout(); fig.savefig(FIG / "real_typing_time_by_rep.png", dpi=150); plt.close(fig)

    med = df.groupby("sessionIndex")["total_DD"].median()
    print("\n[A1] median total DD time by session (all subjects):")
    print("     " + "  ".join(f"S{i}: {v:.3f}" for i, v in med.items()))
    print(f"\nFigures saved to {FIG}")


if __name__ == "__main__":
    main()
