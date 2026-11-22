"""
DS5020 Final Project - Iteration 3: the tiny worked example, step by step
Project 9 (Iteration 3, Q1 and Q2): 4 points in R^2, compute mu_hat, Sigma_hat,
Sigma_hat^{-1} and the Mahalanobis distance of new points, one step at a time.

Data (same as the Iteration 2 hand-worked example):
    x_1 = (1, 2), x_2 = (3, 4), x_3 = (5, 4), x_4 = (3, 6)
New points: (5, 6) and (5, 2).

Steps with no square roots (mean, covariance, LDL^T, inverse, d^2) are done in
exact arithmetic with fractions.Fraction, using the same functions from methods.py
that run the simulation and the real data. Steps that need square roots
(eigenvalues -> W = Sigma^{-1/2}, Cholesky) are done in floating point and
compared with the exact values.

Run from the project root:  python3 src/small_example.py
"""
from fractions import Fraction

import numpy as np
from scipy import stats

from methods import (my_mle, my_centre, my_ldl, my_forward_sub, my_mahalanobis2,
                     my_inverse, my_jacobi_eig, my_inv_sqrt, my_whiten_distance2,
                     my_cholesky, my_threshold, my_decide)
from diagnostics import report, table_row, print_table, header


def show(name, A):
    """Print a vector or matrix of Fractions as exact values."""
    A = np.asarray(A, dtype=object)
    if A.ndim == 1:
        print(f"{name} = (" + ", ".join(str(a) for a in A) + ")")
    else:
        rows = ["[" + ", ".join(f"{str(a):>5}" for a in r) + "]" for r in A]
        print(f"{name} =\n    " + "\n    ".join(rows))


def frac(A):
    return np.array([[Fraction(v) for v in row] for row in A], dtype=object)


X = frac([[1, 2], [3, 4], [5, 4], [3, 6]])
X_new = frac([[5, 6], [5, 2]])
n, d = X.shape

header("Q1, steps 1-3: mean, centring, covariance (exact, Fraction)")
show("X", X)
mu_hat, Sigma_hat = my_mle(X)
print("\nStep 1  mu_hat = (1/n) X^T 1")
print(f"        column sums = ({X[:, 0].sum()}, {X[:, 1].sum()}),  n = {n}")
show("        mu_hat", mu_hat)
X_c = my_centre(X, mu_hat)
print("\nStep 2  X_c = X - 1 mu_hat^T")
show("        X_c", X_c)
print("\nStep 3  Sigma_hat = (1/n) X_c^T X_c")
show("        X_c^T X_c", X_c.T @ X_c)
show("        Sigma_hat", Sigma_hat)

header("Q2: LDL^T factorization and inverse (exact, Fraction)")
L, D = my_ldl(Sigma_hat)
print("D_1 = S_11 = 2;  L_21 = S_21 / D_1 = 1/2;  D_2 = S_22 - L_21^2 D_1 = 2 - 1/2 = 3/2")
show("L", L)
show("D", D)
print("all pivots D_j > 0  =>  Sigma_hat positive definite (Assumption 3)")
show("check L D L^T", L @ (D[:, None] * L.T))
Linv = my_forward_sub(L, frac(np.eye(d, dtype=int)))
show("L^{-1} (forward substitution on I)", Linv)
Sigma_inv = my_inverse(Sigma_hat)
show("Sigma_hat^{-1} = L^{-T} D^{-1} L^{-1}", Sigma_inv)
print("   (expect (1/3) [[2, -1], [-1, 2]])")

header("Steps 7-9: Mahalanobis distance of the new points (exact, Fraction)")
d2_exact = my_mahalanobis2(X_new, mu_hat, L, D)
for x, d2 in zip(X_new, d2_exact):
    v = x - mu_hat
    w = my_forward_sub(L, v[:, None])[:, 0]
    print(f"x = ({x[0]}, {x[1]}):  v = x - mu_hat = ({v[0]}, {v[1]}),"
          f"  L w = v -> w = ({w[0]}, {w[1]}),"
          f"  d^2 = w_1^2/D_1 + w_2^2/D_2 = {w[0] ** 2 / D[0]} + {w[1] ** 2 / D[1]} = {d2}")
eucl = [float(np.sqrt(float(np.sum((x - mu_hat) ** 2)))) for x in X_new]
print(f"Euclidean distances from mu_hat: {eucl[0]:.4f}, {eucl[1]:.4f} (both sqrt 8)")

header("Second-way checks (exact)")
I2 = Sigma_hat @ Sigma_inv
show("Sigma_hat Sigma_hat^{-1}", I2)
ok_inv = all(I2[i, j] == (1 if i == j else 0) for i in range(d) for j in range(d))
d2_direct = np.array([(x - mu_hat) @ Sigma_inv @ (x - mu_hat) for x in X_new], dtype=object)
print("d^2 = v^T Sigma_hat^{-1} v directly:", ", ".join(map(str, d2_direct)),
      "-> equal to the LDL result:", bool(np.all(d2_direct == d2_exact)))
d2_train = my_mahalanobis2(X, mu_hat, L, D)
print("training d^2:", ", ".join(map(str, d2_train)),
      f"-> mean = {d2_train.sum() / n} (Proposition: mean = d = {d})")
table_row("Small example", "Sigma_hat Sigma_hat^-1", "I (exact fractions)", "0", ok_inv)
table_row("Small example", "d^2 (LDL^T)", "v^T Sigma^-1 v (exact)", "0",
          bool(np.all(d2_direct == d2_exact)))
table_row("Small example", "mean training d^2", "d = 2 (exact)",
          str(d2_train.sum() / n - d), d2_train.sum() / n == d)

header("Floating point: eigenvalues (Jacobi), W = Sigma^{-1/2}, Cholesky")
S_f = Sigma_hat.astype(float)
lam, Q = my_jacobi_eig(S_f)
print("Step 4  eigenvalues (Jacobi) =", lam, "  (exact: 1 and 3)")
print("        eigenvectors Q =\n", Q.round(6), "\n        (exact: (1,-1)/sqrt2 and (1,1)/sqrt2, up to sign)")
W = my_inv_sqrt(lam, Q)
print("Step 5  W = Q Lambda^{-1/2} Q^T =\n", W.round(6))
d2_W = my_whiten_distance2(X_new.astype(float), mu_hat.astype(float), W)
Lc = my_cholesky(S_f)
print("Cholesky L_c = L D^{1/2} =\n", Lc.round(6), "  (exact: [[sqrt2, 0], [1/sqrt2, sqrt(3/2)]])")
d2_L = my_mahalanobis2(X_new.astype(float), mu_hat.astype(float), Lc)
print("d^2 via W:", d2_W, "  d^2 via Cholesky:", d2_L)

print()
a1 = report("eigenvalues vs exact (1, 3)", lam, [1, 3])[0]
a2 = report("W^2 vs exact Sigma^-1", W @ W, Sigma_inv.astype(float))[0]
a3 = report("Cholesky vs exact", Lc, [[np.sqrt(2), 0], [1 / np.sqrt(2), np.sqrt(1.5)]])[0]
a4 = report("d^2 (W) vs exact (8/3, 8)", d2_W, d2_exact.astype(float))[0]
a5 = report("d^2 (Cholesky) vs exact", d2_L, d2_exact.astype(float))[0]
a6 = report("eigenvalues vs numpy.eigh", lam, np.linalg.eigvalsh(S_f))[0]
a7 = report("Cholesky vs numpy.cholesky", Lc, np.linalg.cholesky(S_f))[0]
table_row("Small example", "eigenvalues (Jacobi)", "exact (1, 3)", f"max = {a1:.0e}", a1 < 1e-12)
table_row("Small example", "d^2 via W", "exact (8/3, 8)", f"max = {a4:.0e}", a4 < 1e-12)
table_row("Small example", "d^2 via Cholesky", "exact (8/3, 8)", f"max = {a5:.0e}", a5 < 1e-12)

header("Step 6 and 10: threshold and decision (alpha = 0.05)")
tau = my_threshold(0.05, d)
print(f"tau = chi^2_(2, 0.95): ours = {tau:.10f},  -2 ln(0.05) = {-2 * np.log(0.05):.10f},"
      f"  scipy = {stats.chi2.ppf(0.95, 2):.10f}")
for x, d2, flag in zip(X_new, d2_exact, my_decide(d2_exact.astype(float), tau)):
    print(f"x = ({x[0]}, {x[1]}): d^2 = {float(d2):.4f}  ->  {'ANOMALOUS' if flag else 'genuine'}")
e_tau = abs(tau + 2 * np.log(0.05))
table_row("Small example", "tau (alpha = 0.05)", "-2 ln alpha (exact for d=2)", f"{e_tau:.0e}", e_tau < 1e-9)

print_table()
