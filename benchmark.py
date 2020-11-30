from time import time

import autograd.numpy as np

import lab as B

n = 20
m = 1

t = np.float64
eps = B.cast(t, B.epsilon)


def f1(x):
    dists2 = (x - B.transpose(x)) ** 2
    K = B.exp(-0.5 * dists2)
    K = K + B.epsilon * B.eye(t, n)
    L = B.cholesky(K)
    return B.matmul(L, B.ones(t, n, m))


def f2(x):
    dists2 = (x - np.transpose(x)) ** 2
    K = np.exp(-0.5 * dists2)
    K = K + B.epsilon * np.eye(n, dtype=t)
    L = np.linalg.cholesky(K)
    return np.matmul(L, np.ones((n, m)))


# Perform computation once.
x = np.linspace(0, 1, n, dtype=t)[:, None]
f1(x)
f2(x)

its = 10000

s = time()
for _ in range(its):
    z = f2(x)
us_native = (time() - s) / its * 1e6

s = time()
for _ in range(its):
    z = f1(x)
us_lab = (time() - s) / its * 1e6

print(
    "Overhead: {:.1f} us / {:.1f} %"
    "".format(us_lab - us_native, 100 * (us_lab / us_native - 1))
)
