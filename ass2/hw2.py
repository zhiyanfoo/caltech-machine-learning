import os
import sys

above_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, above_dir)

from tools import ProgressIterator, random_target_function, random_set, second_order, pla, linear_percepton, weight_error, output, experiment
import numpy as np

np.random.seed(0)

# HOEFFDING INEQUALITY 

def coin_data(num_trials, sample_size, num_flips):
    return np.random.randint(2, size=(num_trials, sample_size, num_flips))

def collate_flip_results(coin_data):
    return np.sum(coin_data, axis=2)

def experiment_results(collated_flips):
    avg_v1 = np.average(collated_flips[:,0]) / 10
    random_samples = [ trial[np.random.randint(len(trial))] 
            for trial in collated_flips ]
    avg_vrand = np.average(random_samples) / 10
    avg_vmin = np.average(np.amin(collated_flips, axis=1)) / 10
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

def test_four(in_sample, out_sample):
    training_set = random_set(in_sample, moved_circle)
    noisy_indices = np.random.choice(in_sample, size=round(0.1 * in_sample), replace=False)
    training_set.y[noisy_indices] *= -1
    weight = linear_percepton(training_set.z, training_set.y)
    in_error_no_transform = weight_error(weight, training_set.z, training_set.y)
    training_set.z = second_order(training_set.z)
    weight = linear_percepton(training_set.z, training_set.y)
    in_error_transform = weight_error(weight, training_set.z, training_set.y)
    testing_set = random_set(out_sample, moved_circle, second_order)
    noisy_indices = np.random.choice(out_sample, size=round(0.1 * out_sample), replace=False)
    testing_set.y[noisy_indices] *= -1
    out_error_transform = weight_error(weight, testing_set.z, testing_set.y)
    return in_error_no_transform, weight, out_error_transform

def main():
    print("The following simulations are computationally intensive")
    output(simulations)

def simulations():
    que ={}
    progress_iterator = ProgressIterator(4)
    progress_iterator.next()
    avg_v1, avg_vrand, avg_vmin = test_one()
    que[1] = ("v min :", avg_vmin)
    
    progress_iterator.next()
    in_error, out_error = experiment(test_two, [100, 1000], 1000)
    que[5] = ("in sample error :", in_error)
    que[6] = ("out sample error :", out_error)

    progress_iterator.next()
    iterations = experiment(test_three, [10], 1000)
    que[7] = ("iterations :", iterations)

    progress_iterator.next()
    results = np.array([ test_four(100, 1000) for _ in range(1000) ], dtype=object)
    in_error_no_transform = np.mean(results[:,0])
    weight = np.mean(results[:,1], axis=0)
    out_error_transform = np.mean(results[:,2])
    que[8] = ("in sample error -- without higher dimension transformation :",
            in_error_no_transform)
    que[9] = ("higher dimensional weights :", weight)
    que[10] = ("out of sample error -- with higher dimension transformation :",
            out_error_transform)
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
