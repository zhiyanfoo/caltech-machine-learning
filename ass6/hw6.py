import os
import sys

above_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, above_dir)

from tools import *

import numpy as np

np.random.seed(0)

# REGULARIZATION WITH WEIGHT DECAY

def test1(training_data, testing_data):
    training_set = DataML(training_data, transform)
    weight = linear_percepton(training_set.z, training_set.y)
    testing_set = DataML(testing_data, transform)
    in_error, out_error = [ weight_error(weight, data_set.z, data_set.y)
            for data_set in [training_set, testing_set] ]
    return in_error, out_error

def transform(x):
    """
    transform             
    x1 x2  --->   1 x1 x2 x1**2 x2**2 x1x2 |x1 - x2| |x1 + x2|
    """
    ones = np.ones(len(x))
    # print('ones')
    # print(ones)
    x1 = x[:,0]
    x2 = x[:,1]
    x1_sqr = x1**2
    # print('x1_sqr')
    # print(x1_sqr[0])
    x2_sqr = x2**2
    # print('x2_sqr')
    # print(x2_sqr[0])
    x1x2 = x1 * x2
    # print('x1x2')
    # print(x1x2[0])
    abs_x1_minus_x2 = abs(x1-x2)
    # print('abs_x1_minus_x2')
    # print(abs_x1_minus_x2[0])
    abs_x1_plus_x2 = abs(x1+x2)
    # print('abs_x1_plus_x2')
    # print(abs_x1_plus_x2[0])
    return np.stack([ones, x1, x2, x1_sqr, x2_sqr, x1x2, abs_x1_minus_x2, abs_x1_plus_x2], axis=1)

def trial(training_data, testing_data, a):
    training_set = DataML(training_data, transform)
    weights = minimize_error_aug(training_set.z, training_set.y, a)
    in_error, out_error = [ weight_error(weights, tset.z, tset.y)
        for tset in [training_set, DataML(testing_data, transform)] ]
    return in_error, out_error

def minimize_error_aug(z,y,a):
    """
    minimize
    d_Ein = Z(Z*w - y) + a*w = 0
    (Z*Z + a*I)^-1 * Z*y) = w
    Returns: weights
    """
    zz = z.transpose().dot(z)
    zz_plus_ai = zz + a * np.identity(len(zz))
    inv_zz_plus_ai = np.linalg.inv(zz_plus_ai)
    zy = z.transpose().dot(y)
    inv_zz_plus_ai_zy = inv_zz_plus_ai.dot(zy)
    return inv_zz_plus_ai_zy 

def pow_10(k): 
    return 10**k

# REGULARIZATION FOR POLYNOMIALS

def main():
    output(simulations)

def simulations():
    que = {}
    training_data = np.genfromtxt("in.dta")
    testing_data = np.genfromtxt("out.dta")
    in_sample_error, out_of_sample_error = test1(training_data, testing_data)
    que[2] = ("linear regression",
            "\n\tin sample error : " + str(in_sample_error) + \
            "\n\tout of sample error : " + str(out_of_sample_error))
    in_sample_error, out_sample_error = trial(training_data, testing_data, pow_10(-3))
    que[3] = ("linear regression with weight decay, k=-3",
            "\n\tin sample error : " + str(in_sample_error) + \
            "\n\tout of sample error : " + str(out_of_sample_error))
    in_sample_error, out_sample_error = trial(training_data, testing_data, pow_10(3))
    que[4] = ("linear regression with weight decay, k=3",
            "\n\tin sample error : " + str(in_sample_error) + \
            "\n\tout of sample error : " + str(out_of_sample_error))
    out_of_sample_errors = [ str(trial(training_data, testing_data, pow_10(k))[1])
            for k in range(-2,3) ]
    que[5] = ("linear regression with weight decay, k=-2..2",
            "\nout of sample errors\n" + "\n".join(out_of_sample_errors))
    return que

ans = {
        1 : 'b',
        2 : 'a',
        3 : 'd',
        4 : 'e',
        5 : 'd',
        6 : 'b',
        7 : 'c',
        8 : 'd',
        9 : 'a',
        10 : 'e',
        }

if __name__ == "__main__":
    main()

