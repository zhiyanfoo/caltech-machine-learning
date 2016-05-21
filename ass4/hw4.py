import os
import sys

above_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, above_dir)

from tools import ProgressIterator, DataML, get_y, linear_percepton, output
import numpy as np

from math import ceil
from sympy import exp, log, sqrt, power
from sympy import Eq
from sympy import Symbol
from sympy import nsolve
from sympy import plot_implicit
from sympy.plotting import plot
from decimal import Decimal

from scipy.integrate import quad

np.random.seed(0)

# GENERALIZATION ERROR

def solved_vc_inequality(probability, error, approximate_datapoints_needed):
    n = Symbol('n')
    return nsolve(log(4) + 10 * log(2*n) - 1/8 * error ** 2 * n - log(probability), n, approximate_datapoints_needed)

def generate_growth_function_bound(d_vc):
    def growth_function_bound(n):
        return n ** d_vc
    return growth_function_bound

def original_vc_bound(n, delta, growth_function):
    return sqrt(8/n * log(4*growth_function(2*n)/delta))

def rademacher_penalty_bound(n, delta, growth_function):
    return sqrt(2 * log(2 * n * growth_function(n)) / n) + sqrt(2/n * log(1/delta)) + 1/n

def parrondo_van_den_broek_right(error, n, delta, growth_function):
    return sqrt(1/n * (2 * error + log(6 * growth_function(2*n)/delta)))

def devroye(error, n, delta, growth_function):
    return sqrt(1/(2*Decimal(n)) * (4 * error * (1 + error) + log(4 * growth_function(Decimal(n**2))/Decimal(delta))))


def error_bound(n):
    e = Symbol('e')
    y = Symbol('y')
    growth_function_bound = generate_growth_function_bound(50)
    a = original_vc_bound(n, 0.05, growth_function_bound)
    b = nsolve(Eq(rademacher_penalty_bound(n, 0.05, growth_function_bound), y), 1)
    c = nsolve(Eq(parrondo_van_den_broek_right(e, n, 0.05, growth_function_bound), e), 1)
    d = nsolve(Eq(devroye(e, n, 0.05, growth_function_bound), e), 1)
    return a, b, c, d


# def test_three():
#     e = Symbol('e')
#     y = Symbol('y')
#     n = Symbol('n')
#     growth_function_bound = generate_growth_function_bound(50)
#     a = original_vc_bound(5, 0.05, growth_function_bound)
#     b = nsolve(Eq(rademacher_penalty_bound(5, 0.05, growth_function_bound), y), 5)
#     c = nsolve(Eq(parrondo_van_den_broek_right(e, 5, 0.05, growth_function_bound), e), 1)
#     d = nsolve(Eq(devroye(e, 5, 0.05, growth_function_bound), e), 1)
#     return a, b, c, d

def plot():
    e = Symbol('e')
    y = Symbol('y')
    n = Symbol('n')
    generalized_vc_bounds = (original_vc_bound, rademacher_penalty_bound)
    growth_function_bound = generate_growth_function_bound(50)
    p1 = plot(original_vc_bound(n, 0.05, growth_function_bound), (n,100, 15000), show=False, line_color = 'black')
    p2 = plot(rademacher_penalty_bound(n, 0.05, growth_function_bound), (n,100, 15000), show=False, line_color = 'blue')
    plot_implicit(Eq(e, parrondo_van_den_broek_right(e, n, 0.05, growth_function_bound)), (n,100, 15000), (e,0,5))
    # plot_implicit(Eq(e, devroye(e, n, 0.05, growth_function_bound)), (n,100, 1000), (e,0,5))
    p1.extend(p2)
    p1.show()

# BIAS AND VARIANCE 

def sinusoid_over_axis(x):
    return np.sin(np.pi * x[0])

def sinusoid(x):
    return np.sin(np.pi * x)

def random_x_points(n):
    '''
    [[  0.09762701]
     [  0.20552675]
     [ -0.1526904 ]
     [ -0.12482558]
     [  0.92732552]
     [  0.58345008]
     [  0.13608912]
     [ -0.85792788]]
    '''
    return np.random.uniform(-1,1, size=(n,1))

def gen_data(n, target_function, transform=None):
    x = random_x_points(n)
    y = get_y(target_function, x)
    if transform == None:
        return DataML((x, y))
    else:
        return DataML((x, y), transform)

def bias_variance_out_sample_error(n):
    """
    Out of the following hypothesis sets
    h(x) = b
    h(x) = ax
    h(x) = ax + b
    h(x) = ax**2
    h(x) = ax**2 + b
    which has the least out of sample error, when 
    the target function is sin(pi * x).
    The hypothesis error is made out of two components,
    the bias and variance.
    The bias is equal to the following
    bias = E_x[(g_mean(x) - f(x)) ** 2]
    variance = E_x[E_d[(g^(d)(x) - g_mean(x))**2]]
    error = bias plus variance.
    """
    trials = [ gen_data(2, sinusoid_over_axis) for _ in range(n) ]
    hypothesis_functions = (
            (constant, get_constant_parameters),
            (line, get_line_to_parameters),
            (line, get_line_parameters), 
            (quadratic, generate_quadratic_to_parameters), 
            (quadratic, generate_quadratic_parameters),
            )
    analysis  = [ analyse(hypothesis, get_parameters, trials) for hypothesis, get_parameters in hypothesis_functions ]
    # print('argmin', ',', 'min')
    # print(np.argmin(out_of_sample_errors), ',', min(out_of_sample_errors))
    return analysis

def analyse(hypothesis, get_parameters, trials):
    parameters = get_parameters(trials)
    # print('parameters)
    # print(parameters)
    mean_parameters = np.mean(parameters, axis=0)
    # print('mean_parameters')
    # print(mean_parameters)
    bounds = (-1,1)
    bias = expected_bias(hypothesis, sinusoid, bounds, mean_parameters)
    # print('expected_bias')
    # print(bias)
    variance = expected_variance(hypothesis, bounds, parameters, mean_parameters)
    # print('expected_variance')
    # print(variance)
    return {"mean parameters" : mean_parameters,
            "bias" : bias,
            "variance" : variance,
            "expected out of sample error" : bias + variance
            }

def expected_bias(hypothesis, target_function, bounds, mean_parameters):
    sq_error = generate_sq_error(hypothesis, target_function)
    return expected_value_integral(sq_error, bounds, (mean_parameters, ))

def expected_variance(hypothesis, bounds, parameters_list, mean_parameters):
    """ 
    variance = E_x[E_d[(g^(d)(x) - g_mean(x))**2]]
    variance = E_d[E_x[(g^(d)(x) - g_mean(x))**2]]
    """ 
    # print('parameters_list')
    # print(parameters_list)
    individual_variances = [ expected_value_integral(
            generate_sq_error(hypothesis, hypothesis),
            bounds, (parameters, mean_parameters)
        )
            for parameters in parameters_list ]
    # print('individual_variances')
    # print(individual_variances)
    return np.mean(individual_variances)

def expected_value_integral(func, bounds, parameters):
    return quad(func, *bounds, parameters)[0] / (bounds[1] - bounds[0])
    
def generate_sq_error(func1, func2):
    def sq_error(x, func1_parameters=[], func2_parameters=[]):
        # print('func1_parameters')
        # print(func1_parameters)
        # print('func2_parameters')
        # print(func2_parameters)
        return (func1(x, *func1_parameters) - func2(x, *func2_parameters)) ** 2
    return sq_error

def constant(x, c):
    return c

def line(x, gradient, constant=0):
    return gradient * x + constant

def quadratic(x, a, b=0):
    return a * x ** 2 + b

def get_constant_parameters(trials):
    """ put each parameter in a list for uniformity with other parameter functions"""
    return np.array([ [np.mean(training_set.y)] for training_set in trials ])

def get_line_to_parameters(trials):
    """line_to : line through origin"""
    gradients = np.array([ linear_percepton(training_set.x, training_set.y) for training_set in trials ])
    return gradients

def get_line_parameters(trials):
    new_trials = [ 
            DataML((np.insert(training_set.z, 0, 1, axis=1), training_set.y))
        for training_set in trials ]
    weights = np.array([ linear_percepton(training_set.z, training_set.y) for training_set in new_trials ])
    return weights

def generate_quadratic_to_parameters(trials):
    """line_to : line through origin
       function of form ax^2
    """
    def transform(x):
        return x ** 2

    new_trials = [ DataML((training_set.z, training_set.y), transform)
        for training_set in trials ]
    weights = np.array([ linear_percepton(training_set.x, training_set.y) for training_set in new_trials ])
    return weights

def generate_quadratic_parameters(trials):
    """ax^2 + b"""
    def transform(x):
        """
        transform             
        x1  --->   1 x1**2
        """
        ones = np.ones(len(x))
        x1 = x[:, 0]
        x1_sqr = x1 ** 2 
        return np.stack([ones, x1_sqr], axis=1)

    new_trials = [
            DataML((training_set.z, training_set.y), transform)
        for training_set in trials ]
    weights = [ linear_percepton(training_set.z, training_set.y) for training_set in new_trials ]
    return np.array(weights)

def main():
    output(simulations)

def simulations():
    que ={}
    progress_iterator = ProgressIterator(5)
    progress_iterator.next()
    sample_size = ceil(solved_vc_inequality(1 - 0.95, 0.05, 400000))
    que[1] = ("sample size needed :", sample_size)
    def error_bound_format(n):
        original_vc_bound, rademacher_penalty_bound, parrondo_van_den_broek_bound, devroye_bound = error_bound(n)
        output = ("Bounds for N=" + str(n),
            "\noriginal vc : " + str(original_vc_bound)
            + "\n" + "rademacher penalty : " + str(rademacher_penalty_bound)
            + "\n" + "parrondo and van den broek : " + str(parrondo_van_den_broek_bound)
            + "\n" + "devroye : "  + str(devroye_bound)
            + "\n"
            )
        return output

    progress_iterator.next()
    que[2] = error_bound_format(10000)

    progress_iterator.next()
    que[3] = error_bound_format(5) 

    progress_iterator.next()
    analysis = bias_variance_out_sample_error(1000)
    def bias_variance_format(analysis):
        names = [ "constant : a",
                "\n\nline through origin : ax",
                "\n\nline : ax + b",
                "\n\nquadratic through origin : ax**2",
                "\n\nquadratic : ax**2 + b"]
        output = ""
        for i in range(len(analysis)):
            if i == 1:
                output += names[i] \
                        + "\nmean parameters : " + str(analysis[i]["mean parameters"]) + " # ans to question 4 this differs from solution given" \
                        + "\nbias : " + str(analysis[i]["bias"]) + " # ans to question 5" \
                        + "\nvariance : " + str(analysis[i]["variance"]) + " # ans to question 6" \
                        + "\nexpected out of sample error : " + str(analysis[i]["expected out of sample error"])
            else:
                output += names[i] \
                        + "\nmean parameters : " + str(analysis[i]["mean parameters"]) \
                        + "\nbias : " + str(analysis[i]["bias"]) \
                        + "\nvariance : " + str(analysis[i]["variance"]) \
                        + "\nexpected out of sample error : " + str(analysis[i]["expected out of sample error"])
        output += "\n\nbest hypothesis is 'line throgh origin' with an expected out of sample error of " + str(round(analysis[1]["expected out of sample error"], 3))
        return output

    progress_iterator.next()
    que[4] = ("Also includes answers to question 5,6,7\n\nAnalysis of various hypotheses", 
            "\n" + str(bias_variance_format(analysis)))
    return que


ans = {
        1 : 'd',
        2 : 'd',
        3 : 'c',
        4 : 'e?d',
        5 : 'b',
        6 : 'a',
        7 : 'b',
        8 : 'c',
        9 : 'b',
        10 : 'dxe',
        }

if __name__ == '__main__':
    main()
