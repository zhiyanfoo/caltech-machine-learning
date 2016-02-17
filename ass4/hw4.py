import numpy as np
from sympy import exp, log, sqrt, power
from sympy import Eq
from sympy import Symbol
from sympy import nsolve
from sympy import plot_implicit
from sympy.plotting import plot
from decimal import Decimal

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

def question_four():
    a



ans1 = 'd' # 452956.864723099
ans2 = 'c' # computer couldn't handle devroye plotting
ans3 = 'c'
ans4 = ''
ans5 = ''
ans6 = ''
ans7 = ''
ans8 = ''
ans9 = ''
ans10 = ''

def main():
    # print(question_one())
    # question_two()
    # print(question_three())
    print(question_four())
    pass

if __name__ == '__main__':
    main()

