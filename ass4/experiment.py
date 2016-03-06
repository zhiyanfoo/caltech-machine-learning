import numpy as np
from scipy.integrate import quad
# x = np.array([0, 1, 2, 3])
# y = np.array([-1, 0.2, 0.9, 2.1])

# A = np.vstack([x,np.ones(len(x))]).T
# print([x,np.ones(len(x))])
# print(np.vstack([x,np.ones(len(x))]))
# print(A)

# print(np.linalg.lstsq(A,y))


def integrand(x, a, b):
    return a * x + b
c = 2
b = 1
I = quad(integrand, 0, 1, args=(c,b))
print(I)
