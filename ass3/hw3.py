import numpy as np
from scipy.special import comb
def hoeffding_inequality_sample_size_needed(probability, error, num_hypothesis):
    return np.log(probability/2/num_hypothesis)/-2/error**2

def question_one_two_three():
    return hoeffding_inequality_sample_size_needed(0.03, 0.05, 1), hoeffding_inequality_sample_size_needed(0.03, 0.05, 10), hoeffding_inequality_sample_size_needed(0.03, 0.05, 100)

ans1 = 'b'
ans2 = 'c'
ans3 = 'd'
ans4 = 'a'
ans5 = 'e'
ans5 = 'c'

# find the growth function of the two interval hypothesis set
def growth_function_two_interval(n):
    return first_black(n)

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
    return sum([ comb(n-i-1, 2) for i in range(1, n-3+1) ])

def question_seven():
    return [ (growth_function_two_interval(n), growth_function_two_interval_analytical(n)) for n in range(4, 14) ] 


def main():
    # print(question_one_two_three())
    print(question_seven())

if __name__ == '__main__':
    main()


