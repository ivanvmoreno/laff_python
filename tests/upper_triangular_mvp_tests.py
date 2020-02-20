from upper_triangular_mvp import *
from lib.benchmark import benchmark

# Test data
u = np.array([[1, 1, 1, 1], [0, 2, 2, 2], [0, 0, 3, 3], [0, 0, 0, 4]])
x = np.array([1, 2, 3, 4])
y = np.zeros(4, dtype=int)

# Benchmarks
with benchmark('Dot product-based algorithm'):
    mvp_var1(u, x, y)

with benchmark('Axpy-based algorithm'):
    mvp_var2(u, x, y)