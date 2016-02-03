import numpy as np
from scipy.special import comb
def hoeffding_inequality_sample_size_needed(probability, error, num_hypothesis):
    return np.log(probability/2/num_hypothesis)/-2/error**2

def question_one_two_three():
    return hoeffding_inequality_sample_size_needed(0.03, 0.05, 1), hoeffding_inequality_sample_size_needed(0.03, 0.05, 10), hoeffding_inequality_sample_size_needed(0.03, 0.05, 100)


# find the growth function of the two interval hypothesis set
def growth_function_two_interval(n):
    return 2 * (first_black(n) + comb(n-1,2) + n)

def first_black(n):
    total = 0
    for i in range(n-3):
        total += first_white(n-i-1)
    return total

def first_white(n):
    total = 0
    for i in range(n-2):
        total += second_black(n-i-1)
    return total

def second_black(n):
    # total = 0
    # for i in range(n-1):
    #     total += 1
    return n - 1

def growth_function_two_interval_analytical(n):
    return 2 * (sum([ comb(n-i-1, 2) for i in range(1, n-3+1) ]) + comb(n-1,2) + n)

def growth_function_option_a(n):
    return comb(n+1, 4)

def growth_function_option_b(n):
    return comb(n+1, 2) + 1

def growth_function_option_c(n):
    return comb(n+1, 4) + comb(n+1, 2) + 1

def growth_function_option_d(n):
    return comb(n+1, 4) + comb(n+1, 3) + comb(n+1, 2) + comb(n+1, 1) + 1

def question_seven():
    return [ (growth_function_two_interval(n),
        growth_function_two_interval_analytical(n),
        growth_function_option_a(n),
        growth_function_option_b(n),
        growth_function_option_c(n),
        growth_function_option_d(n),
        ) for n in range(4, 14) ] 



def main():
    # print(question_one_two_three())
    print(question_seven())

ans1 = 'b'
ans2 = 'c'
ans3 = 'd'
ans4 = 'a'
ans5 = 'e'
ans5 = 'c'
ans6 = 'c'
ans7 = 'e'
ans8 = 'd'
ans9 = 'a'
ans10 = 'd'

if __name__ == '__main__':
    main()


