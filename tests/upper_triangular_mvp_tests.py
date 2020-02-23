from ..lib.benchmark import benchmark
from ..upper_triangular_mvp import *

# Test data
u = np.array([[1, 1, 1, 1, 1, 1], [0, 2, 2, 2, 2, 2], [0, 0, 3, 3, 3, 3], [0, 0, 0, 4, 4, 4], [0, 0, 0, 0, 5, 5], [0, 0, 0, 0, 0, 6]])
x = np.array([1, 2, 3, 4, 5, 6])
y = np.zeros(6, dtype=int)
yexp = np.array([21, 40, 54, 60, 55, 36])


# Results -
print('Dot product based - Result: %s. Expected: %s' % (mvp_var1(u, x, y), yexp))

# Cleanup
y = np.zeros(6, dtype=int)

print('Axpy based - Result: %s. Expected: %s' % (mvp_var2(u, x, y), yexp))


# Benchmarks -
with benchmark('Dot product-based algorithm'):
    mvp_var1(u, x, y)

# Cleanup
y = np.zeros(6, dtype=int)

with benchmark('uxpy-based algorithm'):
    mvp_var2(u, x, y)