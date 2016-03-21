import sys
sys.path.insert(0, '/Users/zhiyan/Courses/caltech_machine_learning/ass2')
import np_percepton

import numpy as np

data = np.genfromtxt("in.dta")

# print(data)

def linear_percepton(x,y):
    print(x)
    print(y)
    xt_x = x.transpose().dot(x)
    xt_y = x.transpose().dot(y)
    inv_xt_x = np.linalg.inv(xt_x)
    return inv_xt_x.dot(xt_y)

# print(linear_percepton(data[:,0:2],data[:,2]))

def transform(x):
    """
    transform             
    x1 x2  --->   1 x1 x2 x1**2 x2**2 x1x2 |x1 - x2| |x1 + x2|
    """
    print('x')
    print(x[0])
    x1 = x[:,0]
    x2 = x[:,1]
    ones = np.ones(len(x))
    print('ones')
    print(ones)
    x1_sqr = x1**2
    print('x1_sqr')
    print(x1_sqr[0])
    x2_sqr = x2**2
    print('x2_sqr')
    print(x2_sqr[0])
    x1x2 = x1 * x2
    print('x1x2')
    print(x1x2[0])
    abs_x1_minus_x2 = abs(x1-x2)
    print('abs_x1_minus_x2')
    print(abs_x1_minus_x2[0])
    abs_x1_plus_x2 = abs(x1+x2)
    print('abs_x1_plus_x2')
    print(abs_x1_plus_x2[0])
    return np.stack([x1, x2, x1_sqr, x2_sqr, x1x2, abs_x1_minus_x2, abs_x1_plus_x2], axis=1)

newx = transform(data[:,0:2])
print(newx[:,:3])
