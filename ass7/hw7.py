import sys 
sys.path.insert(0, '/Users/zhiyan/Courses/caltech_machine_learning/ass6')
sys.path.insert(0, '/Users/zhiyan/Courses/caltech_machine_learning/ass2')

# sys.setrecursionlimit = 1000

import numpy as np 
from hw6 import DataML
from hw6 import transform, test_weights, linear_percepton

import np_percepton as pct

# import cvxopt
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

from mpmath import mpf

from itertools import chain

# VALIDATION


np.random.seed(10)

def question1to5():
    path = "/Users/zhiyan/Courses/caltech_machine_learning/ass6"
    training_data = np.genfromtxt(path + '/' + "in.dta")
    testing_data = np.genfromtxt(path + '/' + "out.dta")
    inital_total = 25 # initial points used for training
    inital_model_weights = restricted_training(training_data, inital_total)
    print(question1(training_data, inital_total, inital_model_weights))
    question2 = best_model(inital_model_weights, DataML(testing_data, transform))
    print("question2")
    print(question2)
    reverse_total = 10 
    training_set = DataML(training_data[-reverse_total:], transform)
    reverse_model_weights = gen_models(training_set)
    question3 = best_model(reverse_model_weights, DataML(training_data[:-reverse_total], transform))
    print('question3')
    print(question3)
    question4 = best_model(reverse_model_weights, DataML(testing_data, transform))
    print("question4")
    print(question4)

def restricted_training(data, training_total):
    training_set = DataML(data[:training_total], transform)
    model_weights = gen_models(training_set)
    return model_weights


def question1(data, training_total, model_weights):
    testing_set = DataML(data[training_total:], transform)
    print(model_weights)
    return best_model(model_weights, testing_set)

def gen_models(training_set):
    k_values = range(3,8) # k values from question. 8 instead of 7 because range is not inclsuive
    weights = [ linear_percepton(training_set.z[:,:k + 1], training_set.y)
            for k in k_values ] # k+1 as bound is not inclusive
    return weights

def best_model(model_weights, testing_set):
    errors =  [ test_weights(
                    weights, testing_set.z[:,:len(weights)], testing_set.y)
        for weights in model_weights ]
    print("errors")
    print(errors)
    return np.argmin(errors) + 3 # return k value that yields least error. see k_values

def question6():
    def min_e1_e2():
        return min((np.random.random(), np.random.random()))
    e = np.mean([ min_e1_e2() for _ in range(1000) ])
    print("e")
    print(e)

def experiment(in_sample, out_sample, trials_total):
    values = np.array([ trial(in_sample, out_sample) for _ in range(trials_total) ])
    # print(values[:100])
    mean_values = np.mean(values, axis=0)
    print(mean_values)

def question7():
    experiment(10, 200, 1000)
    

def question8():
    experiment(100, 500, 1000)

def classify_data(raw_data, target_function):
    classified = np.array([ target_function(data_point) for data_point in raw_data ])
    return DataML(np.concatenate([raw_data, np.array([classified]).T], axis=1))

def classify_data_linear_binary_random(raw_data):
    m, c = pct.rand_line()
    linear_function = pct.create_linear_function(m, c)
    linear_target_function = pct.create_linear_target_function(linear_function)
    data = classify_data(raw_data, linear_target_function)
    return data, linear_target_function

def binary_percepton(x, y):
    intial_weight = linear_percepton(x, y)
    print('start')
    weight, iterations = update(intial_weight, x, y, 0)
    print(iterations)
    return weight

def update(weight, x, y, iterations):
    mis_point_index = a_misclassified_point(x, y, weight)
    if mis_point_index is None:
        return weight, iterations
    new_weight = weight + x[mis_point_index] * y[mis_point_index]
    # print(new_weight)
    return update(new_weight, x, y, iterations + 1)

def a_misclassified_point(x, y, weight):
    start = np.random.randint(0, len(x) - 1)
    for i in chain(range(start, len(x)), range(start)):
        if pct.sign(np.dot(x[i], weight)) != y[i]:
            return i
    return None

def check_error(x, y, weights):
    classification = np.array([ pct.sign(np.dot(point, weights)) for point in x])
    n_misclassified_points = len(y) - sum(classification == y)
    return n_misclassified_points / len(y)

def trial(in_sample, out_sample):
    raw_data = pct.n_random_datapoint(out_sample)
    data, linear_target_function = classify_data_linear_binary_random(raw_data)
    training_set = DataML(data.z_y[:in_sample])
    testing_set = DataML(data.z_y[in_sample:])
    pla_weight = binary_percepton(training_set.z, training_set.y)
    pla_error = check_error(testing_set.z, testing_set.y, pla_weight)
    # print("pla_weight")
    # print(pla_weight)
    # print("pla_error")
    # print(pla_error)
    svm_weight = svm(training_set.z, training_set.y).flatten()
    # print("svm_weight")
    # print(svm_weight)
    svm_error = check_error(testing_set.z, testing_set.y, svm_weight)
    # print("svm_error")
    # print(svm_error)
    def helper(x):
        if x <= 0:
            return 0
        else:
            return 1
    difference = pla_error - svm_error
    svm_better = helper(difference)
    total_support_vectors = sum([ 1 for x in svm_weight if x >= 10*-3 ])
    return svm_better, total_support_vectors

    
def svm(x, y):
    """
    Minimize
    1/2 * w^T w
    subject to
    y_n (w^T x_n + b) >= 1
    """
    weights_total = len(x[0])
    I_n = np.identity(weights_total-1)
    P_int =  np.vstack(([np.zeros(weights_total-1)], I_n))
    # print("P_int")
    # print(P_int)
    zeros = np.array([np.zeros(weights_total)]).T
    # print("zeros")
    # print(zeros)
    P = np.hstack((zeros, P_int))
    # print("P")
    # print(P)
    q = np.zeros(weights_total)
    # print("q")
    # print(q)
    G = -1 * vec_to_dia(y).dot(x)
    # print("G")
    # print(G)
    h = -1 * np.ones(len(y))
    # print("h")
    # print(h)
    matrix_arg = [ matrix(x) for x in [P,q,G,h] ]
    sol = solvers.qp(*matrix_arg)
    return np.array(sol['x'])
    
def alt_svm(x, y):
    """weights include """
    weights_total = len(x[0])
    P = np.identity(weights_total)
    # print("P")
    # print(P)
    q = np.zeros(weights_total)
    # print("q")
    # print(q)
    G = -1 * vec_to_dia(y).dot(x)
    # print("G")
    # print(G)
    h = -1 * np.ones(len(y))
    # print("h")
    # print(h)
    matrix_arg = [ matrix(x) for x in [P,q,G,h] ]
    sol = solvers.qp(*matrix_arg)
    return np.array(sol['x'])

def vec_to_dia(y):
    dia = [ [ 0 for i in range(i) ] 
            + [y[i]] 
            + [ 0 for i in range(i,len(y) - 1) ]  
            for i in range(len(y)) ]
    return np.array(dia, dtype='d')


    
    

   




    

ans = {
        1 : 'd',
        2 : 'e',
        3 : 'd',
        4 : 'd',
        5 : 'b',
        6 : 'cxd',
        7 : 'c',
        8 : 'c',
        9 : 'e',
        10 : 'b',
        }

def main():
    # question1to5()
    # question6()
    # question7()
    question8()


if __name__ == "__main__":
    main()
