import numpy as np

# Considering matrix a non symmetrized (i.e. elements under diagonal are not stored in memory)

def symmetric_mvp_var1(a, x, y):
    for k, row in enumerate(a):
        y[k] = np.dot(a[:, k][:k], x[:k]) + row[k] * x[k] + np.dot(row[(k + 1):], x[(k + 1):]) + y[k]
    return y

def symmetric_mvp_var2(a, x, y):
    for k in range(len(a)):
        y[:k] = a[:, k][:k] * x[k] + y[:k]
        y[k] = a[:, k][k] * x[k] + y[k]
        y[(k + 1):] = a[k][(k + 1):] * x[k] + y[(k + 1):]
    return y
