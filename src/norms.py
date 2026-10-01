import numpy as np

def L2norm(x):
    s = 0
    for i in x:
        s += i*i
    return np.sqrt(s)

def norm_inf_matrix(A):
    A = np.asarray(A, dtype=float)
    m = 0.0
    for i in range(A.shape[0]):
        row_sum = 0.0
        for j in range(A.shape[1]):
            row_sum += abs(A[i, j])
        if row_sum > m:
            m = row_sum
    return m

def norm_inf_vector(x):
    m = 0.0
    for v in x:
        av = abs(v)
        if av > m:
            m = av
    return m

