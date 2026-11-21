"""
DS5020 Final Project - Iteration 3: the method, implemented from scratch
Project 9: Anomaly Detection with the Multivariate Normal Distribution

Rules followed here:
  * NumPy arrays, matrix products and elementwise operations only.
    numpy.linalg, SciPy and scikit-learn are NOT used in this file; the scripts
    use them only to check our answers.
  * One function per numbered step of the algorithm in the report (Section
    "Algorithm"), with the equation it computes written above it, in the report's
    symbols:  X (n x d), mu_hat, X_c, Sigma_hat, Q, Lambda, W = Sigma_hat^{-1/2},
    v = x - mu_hat, z, d^2, tau, alpha.
  * The same functions run the tiny example (with fractions.Fraction, where the
    step needs no square root), the simulation and the real data.

Two equivalent ways to compute d^2 are provided:
  (a) the Iteration 2 algorithm: Sigma_hat = Q Lambda Q^T (Jacobi eigenvalue
      method), W = Q Lambda^{-1/2} Q^T, z = W v, d^2 = z^T z         [steps 4, 5, 8, 9]
  (b) Cholesky: Sigma_hat = L L^T, solve L z = v by forward substitution,
      d^2 = z^T z                                                     [Iteration 3, Q2]
Both give v^T Sigma_hat^{-1} v; the scripts check that they agree.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Step 0. Log transform:  X_ij = log X^raw_ij
# ---------------------------------------------------------------------------
def my_log_transform(X_raw):
    X_raw = np.asarray(X_raw, dtype=float)
    if np.any(X_raw <= 0):
        raise ValueError("log transform needs strictly positive timings")
    return np.log(X_raw)


# ---------------------------------------------------------------------------
# Steps 1-3. MLEs (Iteration 2, Q1)
#   mu_hat    = (1/n) X^T 1
#   X_c       = X - 1 mu_hat^T
#   Sigma_hat = (1/n) X_c^T X_c          (divide by n, not n - 1)
# Works for float arrays and for object arrays of fractions.Fraction.
# ---------------------------------------------------------------------------
def my_mle(X):
    X = np.asarray(X)
    n = X.shape[0]
    mu_hat = X.sum(axis=0) / n               # step 1
    X_c = X - mu_hat                         # step 2 (broadcasting = 1 mu_hat^T)
    Sigma_hat = (X_c.T @ X_c) / n            # step 3
    return mu_hat, Sigma_hat


def my_centre(X, mu_hat):
    """Step 2 / step 7:  X_c = X - 1 mu_hat^T  (rows v_i = x_i - mu_hat)."""
    return np.asarray(X) - mu_hat


# ---------------------------------------------------------------------------
# Step 4. Eigendecomposition  Sigma_hat = Q Lambda Q^T  (Iteration 2, Q4)
# Cyclic Jacobi method. Each rotation J (angle chosen so that entry (p, q)
# becomes 0) is orthogonal, so A <- J^T A J keeps the eigenvalues, and
# Q <- Q J stays orthogonal. When all off-diagonal entries are ~0,
# A = Lambda and Sigma_hat = Q Lambda Q^T.
#   theta = (a_qq - a_pp) / (2 a_pq),  t = sign(theta) / (|theta| + sqrt(theta^2 + 1)),
#   c = 1 / sqrt(t^2 + 1),  s = t c
# ---------------------------------------------------------------------------
def my_jacobi_eig(S, tol=1e-14, max_sweeps=100):
    A = np.array(S, dtype=float)
    d = A.shape[0]
    if not np.allclose(A, A.T):
        raise ValueError("Jacobi method needs a symmetric matrix")
    Q = np.eye(d)
    scale = np.sqrt(np.sum(A ** 2))
    for _ in range(max_sweeps):
        off = np.sqrt(np.sum((A - np.diag(np.diag(A))) ** 2))
        if off <= tol * scale:
            break
        for p in range(d - 1):
            for q in range(p + 1, d):
                if abs(A[p, q]) <= 1e-300:
                    continue
                theta = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
                if abs(theta) > 1e150:               # a_pq negligible: t ~ 1 / (2 theta)
                    t = 1.0 / (2.0 * theta)
                else:
                    t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + np.sqrt(theta ** 2 + 1.0))
                c = 1.0 / np.sqrt(t ** 2 + 1.0)
                s = t * c
                # A <- A J  (columns p, q)
                Ap, Aq = A[:, p].copy(), A[:, q].copy()
                A[:, p], A[:, q] = c * Ap - s * Aq, s * Ap + c * Aq
                # A <- J^T A  (rows p, q)
                Ap, Aq = A[p, :].copy(), A[q, :].copy()
                A[p, :], A[q, :] = c * Ap - s * Aq, s * Ap + c * Aq
                # Q <- Q J
                Qp, Qq = Q[:, p].copy(), Q[:, q].copy()
                Q[:, p], Q[:, q] = c * Qp - s * Qq, s * Qp + c * Qq
    lam = np.diag(A).copy()
    order = np.argsort(lam)                  # ascending, like numpy.linalg.eigh
    return lam[order], Q[:, order]


# ---------------------------------------------------------------------------
# Step 5. Inverse square root (Iteration 2, Q4)
#   W = Sigma_hat^{-1/2} = Q Lambda^{-1/2} Q^T,  needs lambda_min > 0 (Assumption 3)
# ---------------------------------------------------------------------------
def my_inv_sqrt(lam, Q):
    if lam.min() <= 0:
        raise ValueError(f"Sigma_hat is not positive definite (lambda_min = {lam.min():.3g})")
    return (Q * lam ** -0.5) @ Q.T           # Q diag(lambda^{-1/2}) Q^T


# ---------------------------------------------------------------------------
# Steps 8-9 (Iteration 2 algorithm). Whiten and take the squared length
#   z_i = W v_i,   d^2_i = z_i^T z_i = v_i^T Sigma_hat^{-1} v_i
# In matrix form Z = V W (W is symmetric), d^2_i = sum_j Z_ij^2.
# ---------------------------------------------------------------------------
def my_whiten_distance2(X, mu_hat, W):
    Z = my_centre(X, mu_hat) @ W
    return np.sum(Z ** 2, axis=1)


# ---------------------------------------------------------------------------
# Cholesky route (Iteration 3, Q2)
# LDL^T factorization:  Sigma_hat = L D L^T, L unit lower triangular, D diagonal
#   D_j  = S_jj - sum_{k<j} L_jk^2 D_k
#   L_ij = (S_ij - sum_{k<j} L_ik L_jk D_k) / D_j        (i > j)
# No square roots, so it is exact on fractions. Sigma_hat is positive definite
# exactly when every pivot D_j > 0 (Assumption 3).
# ---------------------------------------------------------------------------
def my_ldl(S, tol=1e-12):
    S = np.asarray(S)
    d = S.shape[0]
    one = S[0, 0] / S[0, 0]                  # 1 of the same type (float or Fraction)
    L = np.zeros((d, d), dtype=S.dtype)
    D = np.zeros(d, dtype=S.dtype)
    for j in range(d):
        L[j, j] = one
        D[j] = S[j, j] - np.sum(L[j, :j] ** 2 * D[:j])
        if not D[j] > tol * abs(S[j, j]):
            raise ValueError(f"Sigma_hat is not positive definite: pivot D_{j + 1} = {float(D[j]):.3g}")
        if j + 1 < d:
            L[j + 1:, j] = (S[j + 1:, j] - L[j + 1:, :j] @ (L[j, :j] * D[:j])) / D[j]
    return L, D


# Cholesky factor:  Sigma_hat = L_c L_c^T  with  L_c = L D^{1/2}  (column j scaled by sqrt(D_j))
def my_cholesky(S):
    L, D = my_ldl(np.asarray(S, dtype=float))
    return L * np.sqrt(D)


# Forward substitution: solve L Z = B for lower-triangular L (B may have many columns)
#   Z_i = (B_i - sum_{k<i} L_ik Z_k) / L_ii,   i = 1, ..., d
def my_forward_sub(L, B):
    L, B = np.asarray(L), np.asarray(B)
    Z = np.zeros(B.shape, dtype=np.result_type(L, B))
    for i in range(L.shape[0]):
        Z[i] = (B[i] - L[i, :i] @ Z[:i]) / L[i, i]
    return Z


# Squared Mahalanobis distance through a triangular factor
#   Cholesky (D = None):  L z = v,  d^2 = z^T z
#   LDL^T   (D given):    L w = v,  d^2 = sum_j w_j^2 / D_j     (exact on fractions)
# Both equal v^T Sigma_hat^{-1} v, because Sigma_hat^{-1} = L^{-T} D^{-1} L^{-1}.
def my_mahalanobis2(X, mu_hat, L, D=None):
    V = my_centre(np.atleast_2d(X), mu_hat).T         # d x m, columns v_i
    Zt = my_forward_sub(L, V)                         # d x m, columns z_i
    if D is None:
        return np.sum(Zt ** 2, axis=0)
    return np.sum(Zt ** 2 / np.asarray(D)[:, None], axis=0)


# Inverse from the LDL^T factors (Iteration 3, Q2)
#   Sigma_hat^{-1} = L^{-T} D^{-1} L^{-1},  L^{-1} from forward substitution on I
def my_inverse(S):
    S = np.asarray(S)
    L, D = my_ldl(S)
    I = np.zeros(S.shape, dtype=S.dtype)
    for j in range(S.shape[0]):
        I[j, j] = S[0, 0] / S[0, 0]          # 1 of the same type (float or Fraction)
    Linv = my_forward_sub(L, I)
    return Linv.T @ (Linv / D[:, None])


# ---------------------------------------------------------------------------
# Step 6. Threshold (my_threshold, my_threshold_F): added in the next commit.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 10. Decide:  flag x as anomalous if d^2(x) > tau
# ---------------------------------------------------------------------------
def my_decide(d2, tau):
    return np.asarray(d2) > tau


# Evaluation:  false-alarm rate = mean(flag | genuine),  detection rate = mean(flag | impostor)
def my_rates(d2_genuine, d2_impostor, tau):
    return float(np.mean(my_decide(d2_genuine, tau))), float(np.mean(my_decide(d2_impostor, tau)))


# ---------------------------------------------------------------------------
# Whole fit + score pipeline (steps 0-10), used by the scripts
# ---------------------------------------------------------------------------
def my_fit(X_train, alpha=0.01, threshold="chi2"):
    """Steps 1-6. Returns a dict with mu_hat, Sigma_hat, Q, Lambda, W, L (Cholesky) and tau."""
    n, d = X_train.shape
    mu_hat, Sigma_hat = my_mle(X_train)
    lam, Q = my_jacobi_eig(Sigma_hat)
    W = my_inv_sqrt(lam, Q)
    L = my_cholesky(Sigma_hat)
    tau = my_threshold(alpha, d) if threshold == "chi2" else my_threshold_F(alpha, d, n)
    return dict(mu_hat=mu_hat, Sigma_hat=Sigma_hat, lam=lam, Q=Q, W=W, L=L, tau=tau,
                n=n, d=d, alpha=alpha)


def my_score(model, X_new):
    """Steps 7-9 (Iteration 2 route, z = W v)."""
    return my_whiten_distance2(X_new, model["mu_hat"], model["W"])
