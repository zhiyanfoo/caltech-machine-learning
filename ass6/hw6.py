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
    errors = question2(training_data, testing_data)
    # print("errors")
    # print(errors)
    # print(question3(training_data, testing_data))
    # print(question4(training_data, testing_data))
    print(question5(training_data, testing_data))

def question2(training_data, testing_data):
    weights, transformed_x = learn(training_data)
    in_error, out_error = [ test_weights(weights, transform(data[:,0:2]), data[:,2])
            for data in [training_data, testing_data] ]
    return in_error, out_error

def test_weights(weights, z, y):
    # print(z)
    # print(y)
    learnt_output = classify(weights, z)
    return classified_error(learnt_output, y)

def classify(weights, z):
    vec_sign = np.vectorize(sign)
    return vec_sign(dot(z, weights))

def classified_error(learnt_output, real_output):
    equality_array = np.equal(learnt_output, real_output)
    return 1 - sum(equality_array) / len(equality_array)

def learn(training_data):
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
    return trial(training_data, testing_data, pow_10(-3))

def trial(training_data, testing_data, a):
    training_set = DataML(training_data, transform)
    # print(training_set)
    weights = minimize_error_aug(training_set.z, training_set.y, a)
    in_error, out_error = [ test_weights(weights, tset.z, tset.y)
        for tset in [training_set, DataML(testing_data, transform)] ]
    return in_error, out_error

class DataML:
    def __init__(self, data, transform=None):
        self.x = data[:,:data.shape[1]-1]
        self.y = data[:,data.shape[1]-1]
        if transform == None:
            self.z = self.x
        else:
            self.z = transform(self.x)

    def __repr__(self):
        z_repr = "input : z\n" + str(self.z)
        y_repr = "output : y\n" + str(self.y)
        return z_repr +"\n" + y_repr

            
def minimize_error_aug(z,y,a):
    """
    minimize
    d_Ein = Z(Z*w - y) + a*w = 0
    (Z*Z + a*I)^-1 * Z*y) = w
    Returns: weights
    """
    zz = z.transpose().dot(z)
    zz_plus_ai = zz + a * np.identity(len(zz))
    inv_zz_plus_ai = inv(zz_plus_ai)
    zy = z.transpose().dot(y)
    inv_zz_plus_ai_zy = inv_zz_plus_ai.dot(zy)
    return inv_zz_plus_ai_zy 

def question4(training_data, testing_data):
    return trial(training_data, testing_data, pow_10(3))

def question5(training_data, testing_data):
    return [ trial(training_data, testing_data, pow_10(k))
        for k in range(-2,3) ]

def pow_10(k): 
    return 10**k

def main():
    question2to6()

if __name__ == "__main__":
    main()

ans = {
        1 : 'b',
        2 : 'a',
        3 : 'd',
        4 : 'e',
        5 : 'd',
        6 : 'b',
        }

