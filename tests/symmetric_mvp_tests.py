from ..symmetric_mvp import *
from ..lib.benchmark import benchmark
from ..symmetric_mvp import *

# Test data
a = np.array([[-1, 2, 4, 1, 0], [2, 0, -1, -2, 1], [4, -1, 3, 1, 2], [1, -2, 1, 4, 3], [0, 1, 2, 3, 2]])
x = np.array([1, 2, 3, 4, 5])
y = np.zeros(5, dtype=int)
yexp = np.array([19, -4, 25, 31, 30])


# Results -
print('Dot product based - Result: %s. Expected: %s' % (symmetric_mvp_var1(a, x, y), yexp))

# Cleanup
y = np.zeros(5, dtype=int)

print('Axpy based - Result: %s. Expected: %s' % (symmetric_mvp_var2(a, x, y), yexp))


# Benchmarks -
with benchmark('Dot product-based algorithm'):
    symmetric_mvp_var1(a, x, y)

# Cleanup
y = np.zeros(5, dtype=int)

with benchmark('Axpy-based algorithm'):
    symmetric_mvp_var2(a, x, y)