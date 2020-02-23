import numpy as np

def mvp_var1(u, x, y):
    for k, row in enumerate(u):
        y[k] = row[k] * x[k] + np.dot(row[(k+1):], x[(k+1):]) + y[k]
    return y

def mvp_var2(u, x, y):
    for k in range(len(u)):
        y[:k] = u[:, k][:k] * x[k] + y[:k]
        y[k] = u[:, k][k] * x[k] + y[k]
    return y
