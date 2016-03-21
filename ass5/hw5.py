import sys
sys.path.insert(0, '/Users/zhiyan/Courses/caltech_machine_learning/ass2')

import np_percepton

import numpy as np

import mpmath as mp
from mpmath import exp, ln
from mpmath import mpf
from mpmath import e

import math

from itertools import cycle
from functools import partial
# from sympy import Eq
# from sympy import Symbol
# from sympy import solve, nsolve
# from sympy import plot_implicit
# from sympy.plotting import plot
# from decimal import Decimal

# from scipy.integrate import quad

np.random.seed(0)
LEARNING_RATE = 0.01

# Linear Regression Error

def question1():
    return datapoints_needed(0.008, 0.1, 8)

    
def datapoints_needed(in_sample_error, variance, dimension):
    return (dimension + 1) * (1 - in_sample_error / variance) ** -1

def visualize():
    import matplotlib.pyplot as plt
    x = np.linspace(0.5,4,100)
    # plt.plot(x, x, 'o', label='Original data', markersize=4)
    plt.plot(x, 2 * x - 1, 'r', markersize=4, color='blue')
    plt.plot(x**0.5, (2 * x - 1)**0.5, 'r', markersize=4)
    plt.axis((0,4,0,4))
    plt.show()

# Gradient Descent

def question5():
    return find_threshold(
            in_error,
            in_error_gradient,
            [1,1],
            0.1,
            mpf(10)**mpf(-14),
            0
            )


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
    print(function(*initial_conditions))
    if function(*initial_conditions) < minimum_value:
        return function(*initial_conditions), initial_conditions, iterations
    next_conditions = [ 
            initial_conditions[i] \
            - learning_rate * gradient(*initial_conditions)[i]
            for i in range(len(initial_conditions))
            ]
    return find_threshold(function, gradient, next_conditions, learning_rate, minimum_value, iterations + 1)

def question6():
    gradient = [in_error_derivative_u, in_error_derivative_v]
    print(find_threshold_coordinate_descent(
            in_error, gradient, [1,1], mpf('0.1'), mpf(10) ** mpf(-1)))
    return coordinate_descent_max_iterations(
        in_error, gradient, [1,1], mpf('0.1'), 30)


def find_threshold_coordinate_descent(
    function, gradient, initial_conditions, learning_rate, minimum_value):
    def descend_ith_dim(ith, conditions, iterations):
        "dim : dimension"
        if function(*conditions) < minimum_value:
            print(minimum_value)
            print(function(*conditions) < minimum_value)
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
            print(function(*conditions))
            return function(*conditions), conditions, iterations
        i = next(ith)
        conditions[i] = conditions[i] \
                - learning_rate * gradient[i](*conditions)
        return descend_ith_dim(ith, conditions, iterations + 1)
    return descend_ith_dim(
            cycle(range(len(gradient))),
            initial_conditions, 0)

def question8():
    return average_trial_results(trial, 100, 1000, 100)

def average_trial_results(trial, in_sample, out_sample, total_trials):
    trials = np.array([ trial(in_sample, out_sample) for _ in range(total_trials) ])
    return np.mean(trials, axis=0)


def trial(in_sample, out_sample):
    training_raw_data = np_percepton.n_random_datapoint(in_sample)
    training_data, target_function = np_percepton.classify_data_linear_binary_random(training_raw_data)
    raw_testing_data = np_percepton.n_random_datapoint(out_sample)
    testing_data = np_percepton.classify_data(raw_testing_data, target_function)
    # training_indices = np.random.choice(out_sample, size=in_sample, replace=False)
    # training_raw = data['raw'][training_indices, :]
    # training_classified = data['classified'][training_indices]
    weights = stochastic_logistic_regression(training_data)
    cross_entrophy_error = average_cross_entrophy_error(testing_data, weights)
    # out_sample_error = test_weights(weights, testing_data)
    print(cross_entrophy_error)
    return weights, cross_entrophy_error

def average_cross_entrophy_error(testing_data, weights):
    error_func = cross_entrophy_error
    total_cross_entrophy_error =  sum(
            [ error_func(
                testing_data['raw'][i],
                testing_data['classified'][i],
                weights)
                for i in range(len(testing_data['raw']))
                ])
    return total_cross_entrophy_error / len(testing_data['raw'])

def test_weights(weights, testing_data):
    total_misclassified = sum([ 1 
        for i in range(len(testing_data['raw']))
        if np_percepton.sign(np.dot(testing_data['raw'][i], weights))
                != testing_data['classified'][i] ])
    print('total_misclassified')
    print(total_misclassified)
    return total_misclassified / len(testing_data['raw'])
    
def stochastic_logistic_regression(training_data):
    gradient = [ partial(cross_entrophy_error_ith_derivative, i=j)
        for j in range(len(training_data['raw'][0])) ]
    weights = np.zeros(len(training_data['raw'][0]))
    func = (cross_entrophy_error, gradient)
    old_run_weights = epoch(training_data, weights, *func)
    new_run_weights = epoch(training_data, old_run_weights, *func)
    i = 0
    while np.linalg.norm(old_run_weights - new_run_weights) > 0.01:
        i += 1
        # print(i)
        old_run_weights = new_run_weights
        new_run_weights = epoch(training_data, old_run_weights, *func)
    return new_run_weights

def cross_entrophy_error(x, y, w):
    return math.log(1+math.exp(-y * np.dot(x,w)))
def cross_entrophy_error_ith_derivative(x, y, w, i):
    return - (y * x[i]) / (1 + math.exp(y * np.dot(x, w)))

def epoch(training_data, weights, error_function, gradient):
    data_index_iter = np.random.permutation(len(training_data['raw']))
    for i in data_index_iter:
        x = training_data['raw'][i]
        y = training_data['classified'][i]
        weights = weights - LEARNING_RATE * np.array(
                    [ derivative(x, y, weights) for derivative in gradient ])
    return weights

ans = {
    1 : 'axc',
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

def main():
    # print(question1())
    # visualize()
    # print(question5())
    # print(question6())
    print(question8())
    pass

def tests():
    # test_in_error_derivative_u()
    # test_in_error_derivative_v()
    # test_in_error_gradient()
    # test_in_error()
    pass

def test_in_error():
    error = in_error(0,1)
    print(error)
    assert approximately_equal(error, 4)

def test_in_error_derivative_u():
    error = in_error_derivative_u(0,1)
    print(error)
    assert approximately_equal(error, 2* ( -2 * (e + 2)))
    
def test_in_error_derivative_v():
    error = in_error_derivative_v(0,1)
    print(error)
    assert approximately_equal(error, 2 * (-2) * (-2))

def test_in_error_gradient():
    gradient = in_error_gradient(0,1)
    results = [ 2* ( -2 * (e + 2)), 2 * (-2) * (-2) ] 
    assert [ approximately_equal(gradient[i], results[i]) for i in range(len(gradient)) ] 

if __name__ == '__main__':
    tests()
    main()
