import os
import sys

above_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, above_dir)

from tools import *
import numpy as np

np.random.seed(0)

# THE PERCEPTRON LEARNING ALGORITHM

def trial(in_sample, out_sample):
    target_function = random_target_function()
    training_set, testing_set = training_testing_set(in_sample, out_sample, target_function)
    initial_weight = np.zeros(len(training_set.x[0]))
    weight, iterations = pla(training_set.z, training_set.y, initial_weight, True)
    out_error = weight_error(weight, testing_set.z, testing_set.y)
    return out_error, iterations

def experiment(in_sample, out_sample, total_trials):
    results = [ trial(in_sample, out_sample) for _ in range(total_trials) ] 
    mean_results = np.mean(results, axis=0)
    return mean_results

def main():
    print(experiment(10, 100, 1000))
    print(experiment(100, 100, 1000))

if __name__ == "__main__":
    main()

ans = {
        7 : 'b',
        8 : 'c',
        9 : 'b',
        10 : 'b',
        }
