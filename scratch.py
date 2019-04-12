import numpy as np
np.random.seed(0)

print(np.random.randn(2,2))

np.random.seed(1)

print(np.random.randn(2,2))

np.random.seed(0)

print(np.random.randn(2,2))