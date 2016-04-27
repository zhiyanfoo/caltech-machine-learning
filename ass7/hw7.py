import os
import sys

tools_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, tools_dir_path)

from tools import *

hw6_dir_path = os.path.join(tools_dir_path,"ass6")
sys.path.insert(0, hw6_dir_path)

import numpy as np 

np.random.seed(0)

# VALIDATION

def restricted_training(data, training_total):
    training_set = DataML(data[:training_total], transform)
    model_weights = gen_models(training_set)
    return model_weights

def gen_models(training_set):
    k_values = range(3,8) # k values from question. 8 instead of 7 because range is not inclsuive
    weights = [ linear_percepton(training_set.z[:,:k + 1], training_set.y)
            for k in k_values ] # k+1 as bound is not inclusive
    return weights

def best_model(model_weights, testing_set):
    errors =  [ weight_error(
                    weight, testing_set.z[:,:len(weight)], testing_set.y)
        for weight in model_weights ]
    return len(model_weights[np.argmin(errors)]) - 1, errors # return k value that yields least error. see k_values

def trial(in_sample, out_of_sample):
    target_function = random_target_function()
    training_set = random_set(in_sample, target_function)
    pla_weight = pla(training_set.z, training_set.y) 
    svm_weight = svm(training_set.z, training_set.y)
    testing_set = random_set(out_of_sample, target_function)
    pla_error, svm_error = [ weight_error(weight, testing_set.z, testing_set.y)
            for weight in
            [ pla_weight, svm_weight] ]
    def helper(x):
        if x > 0:
            return 0
        else:
            return 1
    difference = pla_error - svm_error
    svm_better = helper(difference)
    total_support_vectors = sum([ 1 for x in svm_weight if x >= 10*-3 ])
    return svm_better, total_support_vectors

ans = {
        1 : 'd',
        2 : 'e',
        3 : 'd',
        4 : 'd',
        5 : 'b',
        6 : 'cxd',
        7 : 'c',
        8 : 'c',
        9 : 'd',
        10 : 'b',
        }

def main():
    output(simulations)

def simulations():
    que = {}
    training_data = np.genfromtxt(os.path.join(hw6_dir_path, "in.dta"))
    testing_data = np.genfromtxt(os.path.join(hw6_dir_path, "out.dta"))
    inital_total = 25 # initial points used for training
    inital_model_weights = restricted_training(training_data, inital_total)
    validation_set = DataML(training_data[inital_total:], transform) 
    best_k, out_of_sample_errors = best_model(inital_model_weights, validation_set)
    que[1] = ("validation set out of sample errors, last 10 points",
            "\nerrors : " + str(out_of_sample_errors) \
            + "\nbest k : " + str(best_k)
            )
    testing_set = DataML(testing_data, transform)
    best_k, out_of_sample_errors = best_model(inital_model_weights, testing_set)
    que[2] = ("test set out of sample errors",
            "\nerrors : " + str(out_of_sample_errors) \
            + "\nbest k : " + str(best_k)
            )
    first_error = min(out_of_sample_errors)
    reverse_total = 10 
    training_set = DataML(training_data[-reverse_total:], transform)
    reverse_model_weights = gen_models(training_set)
    best_k, out_of_sample_errors = best_model(reverse_model_weights, DataML(training_data[:-reverse_total], transform))
    que[3] = ("validation set out of sample errors, first 25 points",
            "\nerrors : " + str(out_of_sample_errors) \
            + "\nbest k : " + str(best_k)
            )
    testing_set = DataML(testing_data, transform)
    best_k, out_of_sample_errors = best_model(reverse_model_weights, testing_set)
    que[4] = ("test set out of sample errors",
            "\nerrors : " + str(out_of_sample_errors) \
            + "\nbest k : " + str(best_k)
            )
    second_error = min(out_of_sample_errors)
    que[5] = ("smallest out of sample errors :", str(first_error) + ", " + str(second_error))
    svm_better, total_support_vectors  = experiment(trial, [10, 100], 1000)
    que[8] = ("svm better than pla : ", svm_better)
    svm_better, total_support_vectors  = experiment(trial, [100, 100], 1000)
    que[9] = ("svm better than pla : ", svm_better)
    que[10] = ("total support vectors : ", total_support_vectors)
    return que


if __name__ == "__main__":
    main()
