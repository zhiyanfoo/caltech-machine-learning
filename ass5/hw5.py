import os
import sys

above_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, above_dir)

import mpmath as mp
from mpmath import exp, ln
from mpmath import mpf
from mpmath import e

import math

from tools import random_target_function, random_set, experiment, output
import numpy as np

from itertools import cycle
from functools import partial

np.random.seed(0)

def datapoints_needed(in_sample_error, standard_deviation, dimension):
    return 1 / (1 - in_sample_error / standard_deviation ** 2) * (dimension + 1)

def in_error(u, v):
    return (u*exp(v) - mpf(2)*v*exp(-u)) ** mpf(2)

def in_error_derivative_u(u,v):
    return mpf(2) * (u*exp(v) - mpf(2)*v*exp(-u)) * (exp(v) + mpf(2)*v*exp(-u))

def in_error_derivative_v(u,v):
    return mpf(2) * (u*exp(v) - mpf(2)*v*exp(-u)) * (u*exp(v) - mpf(2)*exp(-u))

def in_error_gradient(u,v):
    return in_error_derivative_u(u,v), in_error_derivative_v(u,v)

def approximately_equal(a, b, c=6):
    return abs(a-b) < 10 ** -c

def find_threshold(function, gradient, initial_conditions, learning_rate, minimum_value, iterations):
    if function(*initial_conditions) < minimum_value:
        return function(*initial_conditions), initial_conditions, iterations
    next_conditions = [ 
            initial_conditions[i] \
            - learning_rate * gradient(*initial_conditions)[i]
            for i in range(len(initial_conditions))
            ]
    return find_threshold(function, gradient, next_conditions, learning_rate, minimum_value, iterations + 1)

def find_threshold_coordinate_descent(
    function, gradient, initial_conditions, learning_rate, minimum_value):
    def descend_ith_dim(ith, conditions, iterations):
        "dim : dimension"
        if function(*conditions) < minimum_value:
            return function(*conditions), conditions, iterations
        i = next(ith)
        conditions[i] = conditions[i] \
                - learning_rate * gradient[i](*conditions)
        return descend_ith_dim(ith, conditions, iterations + 1)
    return descend_ith_dim(
            cycle(range(len(gradient))),
            initial_conditions, 0)

def coordinate_descent_max_iterations(
    function, gradient, initial_conditions, learning_rate, max_iterations):
    def descend_ith_dim(ith, conditions, iterations):
        "dim : dimension"
        if iterations == max_iterations:
            return function(*conditions), conditions, iterations
        i = next(ith)
        conditions[i] = conditions[i] \
                - learning_rate * gradient[i](*conditions)
        return descend_ith_dim(ith, conditions, iterations + 1)
    return descend_ith_dim(
            cycle(range(len(gradient))),
            initial_conditions, 0)

def trial(in_sample, out_sample):
    target_function = random_target_function()
    training_set = random_set(in_sample, target_function)
    weight, iterations = stochastic_logistic_regression(training_set)
    testing_set = random_set(out_sample, target_function)
    out_of_sample_error = error(cross_entrophy_error, testing_set, weight)
    return weight, iterations, out_of_sample_error

def stochastic_logistic_regression(training_set):
    gradient = [ partial(cross_entrophy_error_ith_derivative, i=j)
        for j in range(len(training_set.z[0])) ]
    weight = np.zeros(len(training_set.z[0]))
    func = (cross_entrophy_error, gradient)
    old_run_weight = epoch(training_set, weight, *func)
    new_run_weight = epoch(training_set, old_run_weight, *func)
    i = 0
    while np.linalg.norm(old_run_weight - new_run_weight) > 0.01:
        i += 1
        old_run_weight = new_run_weight
        new_run_weight = epoch(training_set, old_run_weight, *func)
    return new_run_weight, i

def cross_entrophy_error(x, y, w):
    return math.log(1+math.exp(-y * np.dot(x,w)))

def cross_entrophy_error_ith_derivative(x, y, w, i):
    return - (y * x[i]) / (1 + math.exp(y * np.dot(x, w)))

def epoch(training_set, weights, error_function, gradient):
    LEARNING_RATE = 0.01
    data_index_iter = np.random.permutation(len(training_set.z))
    for i in data_index_iter:
        x = training_set.z[i]
        y = training_set.y[i]
        weights = weights - LEARNING_RATE * np.array(
                    [ derivative(x, y, weights) for derivative in gradient ])
    return weights

def error(error_function, testing_set, weights):
    total_error = sum([error_function(testing_set.z[i], testing_set.y[i], weights)
        for i in range(len(testing_set.z)) ])
    mean_error = total_error / len(testing_set.z)
    return mean_error

def main():
    tests()
    print("the following simulations are computationally intensive")
    output(simulations)

def simulations():
    que = {}
    que[1] = ("sample points needed :", datapoints_needed(0.008, 0.1, 8))
    gradient = [in_error_derivative_u, in_error_derivative_v]
    value, point, iterations =  find_threshold(in_error, in_error_gradient, [1,1], 0.1, mpf(10)**mpf(-14), 0 )
    que[5] = ( "gradient descent results", "\n\tvalue : " + str(value) \
                                           + "\n\tpoint : " + str(point) \
                                           + " # ans to question 6" \
                                           + "\n\titerations : " + str(iterations)\
                                           + " # ans to question 5"
                                           )
    gradient = [in_error_derivative_u, in_error_derivative_v]
    value, point, iterations = coordinate_descent_max_iterations(
            in_error, gradient, [1,1], mpf('0.1'), 30)
    que[7] = ( "coordinate gradient descent results", "\n\tvalue : " + str(value) \
                                           + "\n\tpoint : " + str(point) \
                                           + "\n\titerations : " + str(iterations)
                                           )
    def trial_no_weights(*args):
        weight, iterations, out_sample_error = trial(*args)
        return iterations, out_sample_error

    iterations, out_sample_error = experiment(trial_no_weights, [100, 1000], 100)
    que[8] = ("out of sample cross entrophy error :", out_sample_error)
    que[9] = ("iterations :", iterations)
    return que

def tests():
    test_in_error_derivative_u()
    test_in_error_derivative_v()
    test_in_error_gradient()
    test_in_error()

def test_in_error():
    error = in_error(0,1)
    assert approximately_equal(error, 4)

def test_in_error_derivative_u():
    error = in_error_derivative_u(0,1)
    assert approximately_equal(error, 2* ( -2 * (e + 2)))
    
def test_in_error_derivative_v():
    error = in_error_derivative_v(0,1)
    assert approximately_equal(error, 2 * (-2) * (-2))

def test_in_error_gradient():
    gradient = in_error_gradient(0,1)
    results = [ 2* ( -2 * (e + 2)), 2 * (-2) * (-2) ] 
    assert [ approximately_equal(gradient[i], results[i]) for i in range(len(gradient)) ] 

ans = {
    1 : 'c',
    2 : 'd',
    3 : 'c',
    4 : 'e',
    5 : 'd',
    6 : 'e',
    7 : 'a',
    8 : 'd',
    9 : 'a',
    10 : 'bxe'
        }

if __name__ == "__main__":
    main()
