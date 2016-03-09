import sys
sys.path.insert(0, '/Users/zhiyan/Courses/caltech_machine_learning/ass2')

import np_percepton

import numpy as np

from sympy import exp, log, sqrt, power
from sympy import Eq
from sympy import Symbol
from sympy import solve, nsolve
from sympy import plot_implicit
from sympy.plotting import plot
from decimal import Decimal

from scipy.integrate import quad

np.random.seed(0)

def vc_inequality_right_side(growth_function, error, datapoints):
    return 4 * growth_function(2*datapoints) * exp(-1 / 8 * error**2 * datapoints) 

def num_datapoints_needed(probability, growth_function, error, approximate_datapoints_needed):
    n = Symbol('n')
    return nsolve(probability - vc_inequality_right_side(growth_function, error, n), n, approximate_datapoints_needed)


# nsolve can't handle this
# print(num_datapoints_needed(1 - 0.95, lambda x : x ** 10, 0.05, 420000))

def solved_vc_inequality(probability, error, approximate_datapoints_needed):
    n = Symbol('n')
    return nsolve(log(4) + 10 * log(2*n) - 1/8 * error ** 2 * n - log(probability), n, approximate_datapoints_needed)

def question_one():
    return solved_vc_inequality(1 - 0.95, 0.05, 400000)

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

def question_two():
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
    print(nsolve(Eq(devroye(e, 10000, 0.05, growth_function_bound), e), 1))

def question_three():
    e = Symbol('e')
    y = Symbol('y')
    n = Symbol('n')
    growth_function_bound = generate_growth_function_bound(50)
    a = original_vc_bound(5, 0.05, growth_function_bound)
    b = nsolve(Eq(rademacher_penalty_bound(5, 0.05, growth_function_bound), y), 5)
    c = nsolve(Eq(parrondo_van_den_broek_right(e, 5, 0.05, growth_function_bound), e), 1)
    d = nsolve(Eq(devroye(e, 5, 0.05, growth_function_bound), e), 1)
    return a, b, c, d

def target_function(x):
    return np.sin(np.pi * x)

def visualize():
    import matplotlib.pyplot as plt
    x = np.linspace(-1,1,100)
    m = np.mean([ np_percepton.linear_percepton(one_point_data(2)) for i in range(10) ])
    plt.plot(x, target_function(x), 'o', label='Original data', markersize=10)
    plt.plot(x, m*x, 'r', label='Fitted line')
    plt.legend()
    plt.show()
    # x = np.linspace(-15,15,100) # 100 linearly spaced numbers # y = np.sin(x)/x # computing the values of sin(x)/x
    # # compose plot
    # plt.plot(x,y) # sin(x)/x
    # plt.plot(x,y,'co') # same function with cyan dots
    # plt.plot(x,2*y,x,3*y) # 2*sin(x)/x and 3*sin(x)/x
    # plt.show() # show the plot

def one_point_data(n):
    data = { 'raw' : np.random.uniform(-1,1, size=(n,1)) }
    data['classified'] = target_function(data['raw'])
    return data

def question_four():
    # visualize()
    # print(one_point_data(2))
    return np.mean([ np_percepton.linear_percepton(one_point_data(2)) for i in range(100000) ])

def question_five():
    m = np.mean([ np_percepton.linear_percepton(one_point_data(2)) for i in range(100000) ])
    def integrand(x):
        return (m * x - target_function(x)) ** 2
    bounds = (-1, 1)
    return quad(integrand,*bounds)[0] / (bounds[1] - bounds[0])


def question_six():
    trials = [ one_point_data(2) for _ in range(10) ]
    gradients = [ np_percepton.linear_percepton(data) for data in trials ]
    mean_gradient = np.mean(gradients)
    bounds = (-1, 1)
    error = [ quad(lambda x : ( (np_percepton.linear_percepton(data) - mean_gradient) * x ) ** 2, *bounds)[0] / (bounds[1] - bounds[0])
        for data in trials ]
    return np.mean(error)
    
# def question_seven():
#     hypothesess_parameters_generators = ( generate_constant_parameters, generate_line_parameters, generate_linear_parameters, generate_ax_sqr_parameters, generate_quadratic_parameters)
#     hypothesess = ( constant, line, linear, ax_sqr, quadratic)
    # trials = [ one_point_data(2) for _ in range(10) ]
#     hypothesess_parameters = [ generate_hypothesis_parameters(trials) for generate_hypothesis_parameters in hypothesess_parameters_generators ]
#     print(hypothesess_parameters)
#     hypothesess_error_functions = [ generate_error_sq(hypothesis, target_function) 
#             for hypothesis in hypothesess]
#     hypotheses_out_of_sample_error = [ 
#             quad(hypothesess_error_functions[i], -1, 1, hypothesess_parameters[i]) 
#             for i in range(len(hypothesess_error_functions)) ]
#     print('hypotheses_out_of_sample_error values')
#     print([ sample_error[0] for sample_error in hypotheses_out_of_sample_error ])
#     return min(hypotheses_out_of_sample_error)

def questions7_least_error():
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
    trials = [ one_point_data(2) for _ in range(1000) ]
    print('trials')
    print(trials)
    hypothesis_functions = (
            # (constant, get_constant_parameters),
            (line, get_line_to_parameters),
            # (line, get_line_parameters), 
            # (quadratic, generate_quadratic_to_parameters), 
            # (quadratic, generate_quadratic_parameters),
            )
    out_of_sample_errors = [ get_expected_out_of_sample_error(hypothesis, get_parameters, trials) for hypothesis, get_parameters in hypothesis_functions ]
    print('out_of_sample_errors')
    print(out_of_sample_errors)
    print('argmin', ',', 'min')
    print(np.argmin(out_of_sample_errors), ',', min(out_of_sample_errors))
    # return get_expected_out_of_sample_error(line, get_line_parameters, trials)

def get_expected_out_of_sample_error(hypothesis, get_parameters, trials):
    parameters = get_parameters(trials)
    print('parameters')
    print(parameters)
    mean_parameters = np.mean(parameters, axis=0)
    print('mean_parameters')
    print(mean_parameters)
    bounds = (-1,1)
    bias = expected_bias(hypothesis, target_function, bounds, mean_parameters)
    print('expected_bias')
    print(bias)
    variance = expected_variance(hypothesis, target_function, bounds, parameters, mean_parameters)
    print('expected_variance')
    print(variance)
    return bias + variance

def expected_bias(hypothesis, target_function, bounds, mean_parameters):
    sq_error = generate_sq_error(hypothesis, target_function)
    return expected_value_integral(sq_error, bounds, (mean_parameters, ))

def expected_variance(hypothesis, target_function, bounds, parameters_list, mean_parameters):
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

def convert_linear_percepton_result_to_parameters(weights):
    return [ [ unneeded_list[0] for unneeded_list in list_of_unneeded_lists ] 
                for list_of_unneeded_lists in weights ]

def constant(x, c):
    return c

def line(x, gradient, constant=0):
    return gradient * x + constant

def quadratic(x, a, b=0):
    return a * x ** 2 + b

def get_constant_parameters(trials):
    """ put each paramter in a list for uniformity with other parameter functions """
    return [ [np.mean(data['classified'])] for data in trials ]

def get_line_to_parameters(trials):
    """line_to : line through origin"""
    int_gradients = [ np_percepton.linear_percepton(data) for data in trials ]
    print(int_gradients)
    gradients = convert_linear_percepton_result_to_parameters(int_gradients)
    print('gradients')
    print(np.array(gradients))
    return np.array(gradients)

def get_line_parameters(trials):
    new_trials = [ { 'classified' : data['classified'], 
        'raw' : np.insert(data['raw'], 1, 1, axis=1) }
        for data in trials ]
    # print(new_trials[0]['raw'])
    int_weights = [ np_percepton.linear_percepton(data) for data in new_trials ]
    print('int_weights')
    print(int_weights)
    weights = convert_linear_percepton_result_to_parameters(int_weights)
    print('weights')
    print(np.array(weights))
    return np.array(weights)

def generate_quadratic_to_parameters(trials):
    """line_to : line through origin
       function of form ax^2
    """
    new_trials = [ { 'classified' : data['classified'], 
        'raw' : data['raw'] ** 2 }
        for data in trials ]
    int_weights = [ np_percepton.linear_percepton(data) for data in new_trials ]
    weights = convert_linear_percepton_result_to_parameters(int_weights)
    print('weights')
    print(weights)
    return np.array(weights)

def generate_quadratic_parameters(trials):
    """ax^2 + b"""
    new_trials = [ { 'classified' : data['classified'], 
        'raw' : np.insert(data['raw'] ** 2, 1, 1, axis=1) }
        for data in trials ]
    int_weights = [ np_percepton.linear_percepton(data) for data in new_trials ]
    print('int_weights')
    print(int_weights)
    weights = convert_linear_percepton_result_to_parameters(int_weights)
    print('weights')
    print(weights)
    return np.array(weights)

def proof_that_linear_percepton_works():
    trials = [one_point_data(2) for i in range(10)]
    np_processed_trials = [ np.linalg.lstsq(data['raw'], data['classified']) for data in trials ]
    home_processed_trials = [ np_percepton.linear_percepton(data) for data in trials ]
    print(np.array(np_processed_trials)[:,0])
    print(home_processed_trials)
    # print(np.linalg.lstsq(trials['raw'], trials['classified']))

def calculus_weights(data):
    x = data['raw']
    y = data['classified']
    sum_of_xi_yi = sum([ x[i] * y[i] for i in range(len(x)) ])
    sum_of_xi_sqr = sum([ xi ** 2 for xi in x ])
    return sum_of_xi_yi / sum_of_xi_sqr 

def np_weights(data):
    x = data['raw']
    y = data['classified']
    return np.linalg.lstsq(x,y)


ans1 = 'd' # 452956.864723099
ans2 = 'c' # computer couldn't handle devroye plotting
ans3 = 'c'
ans4 = 'e'
ans5 = 'b'
ans6 = 'a'
ans7 = 'c'
ans8 = 'c'
ans9 = 'b'
ans10 = 'dxe'

def main():
    # print(question_one())
    # question_two()
    # print(question_three())
    # print(question_four())
    # print(question_five())
    # print(question_six())
    # print(question_seven())
    # print(questions7_least_error())
    print(question8_find_breakpoint())
    pass

if __name__ == '__main__':
    main()


    # print('weights')
    # print(weights)
    # import matplotlib.pyplot as plt

    # v = np.linspace(-1,1,100)
    # x = trials[4]['raw']
    # print('x')
    # print(x)
    # y = trials[4]['classified']
    # print('y')
    # print(y)
    # plt.plot(x, y, 'o', label='Original data', markersize=10)
    # plt.plot(v, weights[4][0][0]*v + weights[4][1][0], 'r', label='lstsqu')
    # plt.legend()
    # plt.show()

    # import matplotlib.pyplot as plt

    # v = np.linspace(-1,1,100)
    # x = trials[4]['raw']
    # y = trials[4]['classified']
    # plt.plot(x, y, 'o', label='Original data', markersize=10)
    # plt.plot(v, gradients[4][0][0]*v, 'r', label='lstsqu')
    # plt.plot(v, alt_gradients[4]*v, 'r', label='calc', color='g')
    # plt.legend()
    # plt.show()

    # print('gradients')
    # print(gradients)
    # print('alt_gradients')
    # print(alt_gradients)
    # print('np_weights')
    # print(np_gradients)

    # import matplotlib.pyplot as plt
    # x = np.linspace(-1,1,100)
    # plt.plot(x, target_function(x), 'o', label='Original data', markersize=1)
    # plt.plot(x, hypothesess_parameters[1]*x, 'r', label='line')
    # plt.plot(x, hypothesess_parameters[2][0]*x, 'r', label='linear', color='g')
    # # plt.legend()
    # plt.show()
