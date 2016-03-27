import sys 
sys.path.insert(0, '/Users/zhiyan/Courses/caltech_machine_learning/ass2')
import np_percepton 
from np_percepton import sign

import numpy as np
from numpy import dot

from numpy.linalg import inv

# REGULARIZATION WITH WEIGHT DECAY

def question2to6():
    training_data = np.genfromtxt("in.dta")
    testing_data = np.genfromtxt("out.dta")
    # errors = question2(training_data, testing_data)
    question3(training_data, testing_data)

def question2(training_data, testing_data):
    weights, transformed_x = learning(training_data)
    in_learnt_output = classify(weights, transformed_x)
    in_error = classified_error(in_learnt_output, training_data[:,2])
    out_learnt_output = classify(weights, transform(testing_data[:,0:2]))
    out_error = classified_error(out_learnt_output, testing_data[:,2])
    return in_error, out_error

def classify(weights, x):
    vec_sign = np.vectorize(sign)
    return vec_sign(dot(x, weights))

def classified_error(learnt_output, real_output):
    equality_array = np.equal(learnt_output, real_output)
    return 1 - sum(equality_array) / len(equality_array)

def learning(training_data):
    transformed_x = transform(training_data[:,0:2])
    return linear_percepton(transformed_x, training_data[:,2]), transformed_x

def linear_percepton(x,y):
    # print(x)
    # print(y)
    xt_x = x.transpose().dot(x)
    xt_y = x.transpose().dot(y)
    inv_xt_x = np.linalg.inv(xt_x)
    return inv_xt_x.dot(xt_y)

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

def question3(training_data, testing_data):
    learning_aug_error(training_data, 10**-3)
    return 

def learning_aug_error(training_data, a):
    transformed_x = transform(training_data[:,0:2])
    return minimize_error_aug(transformed_x, training_data[:,2], a), transformed_x

def minimize_error_aug(z,y,a):
    """
    minimize
    d_Ein = Z(Z*w - y) + a*w = 0
    (Z*Z + a*I)^-1 * Z*y) = w
    Returns: weights
    """
    # print(z.shape)
    zz = z.transpose().dot(z)
    print(zz.shape)
    zz_plus_ai = zz + a * np.identity(len(zz))
    # print("zz_plus_ai")
    # print(zz_plus_ai.shape)
    inv_zz_plus_ai = inv(zz_plus_ai)
    print(z)
    print(y)
    print(z.shape)
    print(y.shape)
    zy = z.transpose().dot(y)
    inv_zz_plus_ai_zy = inv_zz_plus_ai.dot(zy)
    print(inv_zz_plus_ai_zy)
    return inv_zz_plus_ai_zy 


def main():
    question2to6()

if __name__ == "__main__":
    main()

ans = {
        1 : 'b',
        2 : 'a',
        }

