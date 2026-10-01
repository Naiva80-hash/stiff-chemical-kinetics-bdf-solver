import numpy as np
from norms import norm_inf_matrix
def lu_factor_fast(A, pivot_tol=1e-12):
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError(f"lu_factor_fast expects square matrix, got {A.shape}")

    p = np.arange(n)

    scale = norm_inf_matrix(A)
    tiny = pivot_tol * max(1.0, scale)

    for k in range(n - 1):
        # pivot
        imax = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[imax, k]) < tiny:
            raise ValueError(f"Singular/ill-conditioned pivot at col {k}")

        if imax != k:
            A[[k, imax], :] = A[[imax, k], :]
            p[[k, imax]] = p[[imax, k]]

        # elimination (vectorized)
        A[k+1:, k] /= A[k, k]
        A[k+1:, k+1:] -= np.outer(A[k+1:, k], A[k, k+1:])

    if abs(A[-1, -1]) < tiny:
        raise ValueError(f"Singular/ill-conditioned pivot at last col {n-1}")

    return A, p


def lu_solve_fast(LU, p, b):
    b = np.asarray(b, dtype=float)
    vec = (b.ndim == 1)
    if vec:
        b = b.reshape(-1, 1)

    n = LU.shape[0]
    if b.shape[0] != n:
        raise ValueError("Incompatible b shape")

    # apply permutation: Pb
    y = b[p, :].copy()

    # forward solve Ly = Pb  
    for i in range(n):
        y[i, :] -= LU[i, :i] @ y[:i, :]

    # back solve Ux = y 
    x = y.copy()
    for i in range(n - 1, -1, -1):
        x[i, :] -= LU[i, i+1:] @ x[i+1:, :]
        x[i, :] /= LU[i, i]

    return x.ravel() if vec else x
