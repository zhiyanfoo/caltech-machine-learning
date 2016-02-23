import sys
sys.path.insert(0, '/Users/zhiyan/Courses/caltech_machine_learning/ass2')

import np_percepton

import numpy as np

from sympy import exp, log, sqrt, power
from sympy import Eq
from sympy import Symbol
from sympy import nsolve
from sympy import plot_implicit
from sympy.plotting import plot
from decimal import Decimal

from scipy.integrate import quad

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
    return quad(integrand, -1, 1) 


def question_six():
    trials = [ one_point_data(2) for _ in range(10000) ]
    gradients = [ np_percepton.linear_percepton(data) for data in trials ]
    mean_gradient = np.mean(gradients)
    return np.mean([ quad(lambda x : ( (np_percepton.linear_percepton(data) - mean_gradient) * x ) ** 2, -1, 1)
        for data in trials ])
    
def out_of_sample_error(hypthesis):
    pass

def proof_that_linear_percepton_works():
    trials = [one_point_data(2) for i in range(10)]
    np_processed_trials = [ np.linalg.lstsq(data['raw'], data['classified']) for data in trials ]
    home_processed_trials = [ np_percepton.linear_percepton(data) for data in trials ]
    print(np.array(np_processed_trials)[:,0])
    print(home_processed_trials)
    # print(np.linalg.lstsq(trials['raw'], trials['classified']))

ans1 = 'd' # 452956.864723099
ans2 = 'c' # computer couldn't handle devroye plotting
ans3 = 'c'
ans4 = 'e'
ans5 = 'c'
ans6 = 'a'
ans7 = ''
ans8 = ''
ans9 = ''
ans10 = ''

def main():
    # print(question_one())
    # question_two()
    # print(question_three())
    # print(question_four())
    # print(question_five())
    print(question_six())
    pass

if __name__ == '__main__':
    main()

