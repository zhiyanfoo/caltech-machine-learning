import os
import sys

above_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, above_dir)

from tools import random_target_function, random_set, pla, weight_error, output, experiment
import numpy as np

np.random.seed(0)

# THE PERCEPTRON LEARNING ALGORITHM

def trial(in_sample, out_sample):
    target_function = random_target_function()
    training_set = random_set(in_sample, target_function)
    initial_weight = np.zeros(len(training_set.x[0]))
    weight, iterations = pla(training_set.z, training_set.y, initial_weight, True)
    testing_set = random_set(out_sample, target_function)
    out_error = weight_error(weight, testing_set.z, testing_set.y)
    return out_error, iterations

def main():
    output(simulations)

def simulations():
    que ={}
    out_error, iterations = experiment(trial, [10, 100], 1000)
    que[7] = ("iterations :", iterations)
    que[8] = ("out of sample error :", out_error)
    out_error, iterations = experiment(trial, [100, 100], 1000)
    que[9] = ("iterations :", iterations)
    que[10] = ("out of sample error :", out_error)
    return que

ans = {
        1 : 'axd',
        2 : 'a',
        3 : 'd',
        4 : 'b',
        5 : 'c',
        6 : 'e',
        7 : 'b',
        8 : 'c',
        9 : 'b',
        10 : 'b',
        }

if __name__ == "__main__":
    main()
