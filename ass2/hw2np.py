import os
import sys

above_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, above_dir)

from tools import *
import numpy as np
import np_percepton as perc

np.random.seed(0)

# HOEFFDING INEQUALITY 

def coin_data(num_trials, sample_size, num_flips):
    return np.random.randint(2, size=(num_trials, sample_size, num_flips))

def collate_flip_results(coin_data):
    return np.sum(coin_data, axis=2)

def experiment_results(collated_flips):
    avg_v1 = np.average(collated_flips[:,0])
    random_samples = [ trial[np.random.randint(len(trial))] 
            for trial in collated_flips ]
    avg_vrand = np.average(random_samples)
    avg_vmin = np.average(np.amin(collated_flips, axis=1))
    return avg_v1, avg_vrand, avg_vmin

def test_one():
    data = coin_data(100000,1000,10)
    col_results = collate_flip_results(data)
    return experiment_results(col_results)

# LINEAR REGRESSION

def test_two(in_sample, out_sample):
    target_function = random_target_function()
    training_set = random_set(in_sample, target_function)
    weight = linear_percepton(training_set.z, training_set.y)
    in_error = weight_error(weight, training_set.z, training_set.y)
    testing_set = random_set(out_sample, target_function)
    out_error = weight_error(weight, testing_set.z, testing_set.y)
    return in_error, out_error

def test_three(in_sample):
    target_function = random_target_function()
    training_set = random_set(in_sample, target_function)
    return pla(training_set.z, training_set.y, return_iterations=True)[1]


# NONLINEAR TRANSFORMATION

def moved_circle(data_point):
    if data_point[1] ** 2 + data_point[2] ** 2 - 0.6 < 0:
        return -1
    else:
        return 1

def test_four(in_sample):
    training_set = random_set(in_sample, moved_circle)
    noisy_indices = np.random.choice(in_sample, size=0.1 * in_sample, replace=False)
    print('noisy_indices')
    print(noisy_indices)
    training_set.z[noisy_indices] *= -1
    # print(training_set.z == training_set.x)
    print(training_set.z[noisy_indices] *= -1 == training_set.z[noisy_indices])

def part_five(in_sample, out_sample):
    raw_data = perc.n_random_datapoint(out_sample)
    intermediate_data = perc.classify_data(raw_data, perc.non_linear_target_function)
    # print('intermediate_data')
    # print(intermediate_data)
    # xy, x^2, y^2
    xy = np.array([ vector[1] * vector[2] 
            for vector in intermediate_data['raw'] ])
    # print('xy')
    # print(xy)
    xsquared = np.array([ vector[1] ** 2
            for vector in intermediate_data['raw'] ])
    # print('xsquared')
    # print(xsquared)
    ysquared = np.array([ vector[2] ** 2
            for vector in intermediate_data['raw'] ])
    # print('ysquared')
    # print(ysquared)
    additional_data = np.array([xy, xsquared, ysquared]).T
    # print('additional_data')
    # print(additional_data)
    data_raw = np.concatenate((intermediate_data['raw'], additional_data), axis=1)
    # print('data_raw')
    # print(data_raw)
    data = { 'raw': data_raw, 'classified': intermediate_data['classified'] }
    # print('data')
    # print(data)

    noisy_indices = np.random.choice(out_sample, size=0.1 * out_sample, replace=False)
    data['classified'][noisy_indices] *= -1
    training_indices = np.random.choice(out_sample, size=in_sample, replace=False)
    training_raw = data['raw'][training_indices, :]
    training_classified = data['classified'][training_indices]
    training_data = { 'raw' : training_raw, 'classified' : training_classified } 
    linear_weight = perc.linear_percepton(training_data)
    return perc.check_error(training_data, linear_weight), perc.check_error(data, linear_weight), linear_weight

def lab(func, num_trials, *func_args):
    trials = np.array([ func(*func_args) for _ in range(num_trials) ])
    # print(trials)
    return np.mean(trials, axis=0)

def lab_two(func, num_trials, *func_args):
    trials = [ func(*func_args) for _ in range(num_trials) ]
    errors = np.array([ (vector[0], vector[1]) for vector in trials ])
    linear_weights = np.array([ vector[2] for vector in trials ])
    # print('trials')
    # print(trials)
    # print('errors')
    # print(errors)
    # print('linear_weights')
    # print(linear_weights)
    return np.mean(errors, axis=0), np.mean(linear_weights, axis=0)

def main():
    output(simulations)

def simulations():
    que ={}
    # avg_v1, avg_vrand, avg_vmin = test_one()
    # que[1] = ("v min :", avg_vmin)
    # in_error, out_error = experiment(test_two, [100, 1000], 1000)
    # que[5] = ("in sample error :", in_error)
    # que[6] = ("out sample error :", out_error)
    # iterations = experiment(test_three, [10], 1000)
    # que[7] = ("iterations :", iterations)
    test_four(100)
    return que


ans = {
        1 : 'b',
        2 : 'd',
        3 : 'e',
        4 : 'b',
        5 : 'c',
        6 : 'c',
        7 : 'a',
        8 : 'd',
        9 : 'a',
        10 : 'b',
        }

if __name__ == '__main__':
    main()
