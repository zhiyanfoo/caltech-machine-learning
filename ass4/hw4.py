import numpy as np
from sympy import exp, log, sqrt, power
from sympy import Eq
from sympy import Symbol
from sympy import nsolve
from sympy import plot_implicit
from sympy.plotting import plot

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

def original_vc_bound(n, delta):
    return sqrt(8/n * log(4*(2*n)**50/delta))

def rademacher_penalty_bound():
    pass

def parrondo_van_den_broek():
    pass

def devroye():
    pass

def question_two():
    x = Symbol('x')
    y = Symbol('y')
    n = Symbol('n')
    growth_function_bound = generate_growth_function_bound(50)
    # plot(sqrt(8/n * log(4*(2*n)**50)/0.05), (n)
    plot(sqrt(8/n * log(4*(2*n)**50)/0.05), (n,100, 15000))
    # plot(x**2)
    # plot_implicit(Eq(x - y, 1))

# p8 = plot_implicit(y - 1, y_var=y)
# p9 = plot_implicit(x - 1, x_var=x)

ans1 = 'd' # 452956.864723099
ans2 = ''
ans3 = ''
ans4 = ''
ans5 = ''
ans6 = ''
ans7 = ''
ans8 = ''
ans9 = ''
ans10 = ''

def main():
    # print(question_one())
    question_two()
    pass

if __name__ == '__main__':
    main()

