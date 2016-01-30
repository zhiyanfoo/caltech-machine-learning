import numpy as np
import np_percepton as perc

np.random.seed(0)
# Hoeffding Inequality 
# question 1,2

def experiment(num_trials, sample_size, num_flips):
    return np.random.randint(2, size=(num_trials, sample_size, num_flips))

def collate_flip_results(experiment):
    return np.sum(experiment, axis=2)

def experiment_results(experiment_collated_flips):
    avg_v1 = np.average(experiment_collated_flips[:,0])
    random_samples = [ trial[np.random.randint(len(trial))] 
            for trial in experiment_collated_flips ]
    avg_vrand = np.average(random_samples)
    avg_vmin = np.average(np.amin(experiment_collated_flips, axis=1))
    return avg_v1, avg_vrand, avg_vmin

def part_one():
    exp = experiment(100000,1000,10)
    col_exp = collate_flip_results(exp)
    # print('exp\n', exp)
    # print('exp sum\n', col_exp)
    print('avg1, avgrand, avgmin', experiment_results(col_exp))

ans1 = 'b'
ans2 = 'd'
# ____________________________________________________________________________

# Linear Regression
# question 5, 6, 7



def part_two():
    # trial_results = perc.trial(100, 1000)
    # print(trial_results)
    print(perc.average_trial_results(1000, 100, 1000))

def part_three():
    raw_data = perc.n_random_datapoint(10)
    data, target_function = perc.classify_data_linear_binary_random(raw_data)
    return perc.binary_percepton(data)[1]

ans5 = 'b'
ans6 = 'd'
ans7 = 'd'

# Nonlinear Transformation
# question 8, 9, 10

def part_four(in_sample, out_sample):
    raw_data = perc.n_random_datapoint(out_sample)
    data = perc.classify_data(raw_data, perc.non_linear_target_function)
    noisy_indices = np.random.choice(out_sample, size=0.1 * out_sample, replace=False)
    # print("data['classified']")
    # print(data['classified'])
    # print('noisy_indices')
    # print(noisy_indices)
    data['classified'][noisy_indices] *= -1
    training_indices = np.random.choice(out_sample, size=in_sample, replace=False)
    training_raw = data['raw'][training_indices, :]
    training_classified = data['classified'][training_indices]
    training_data = { 'raw' : training_raw, 'classified' : training_classified } 
    linear_weight = perc.linear_percepton(training_data)
    return perc.check_error(training_data, linear_weight), perc.check_error(data, linear_weight) 

def part_five():
    raw_data = perc.n_random_datapoint(out_sample)
    data = perc.classify_data(raw_data, perc.non_linear_target_function)
    noisy_indices = np.random.choice(out_sample, size=0.1 * out_sample, replace=False)
    data['classified'][noisy_indices] *= -1
    training_indices = np.random.choice(out_sample, size=in_sample, replace=False)
    training_raw = data['raw'][training_indices, :]
    training_classified = data['classified'][training_indices]
    training_data = { 'raw' : training_raw, 'classified' : training_classified } 
    pass


ans8 = 'd'

def lab(func, num_trials, *func_args):
    trials = np.array([ func(*func_args) for _ in range(num_trials) ])
    # print(trials)
    return np.mean(trials, axis=0)

def main():
    # part_two()
    # print(lab(part_three, 1000))
    print(lab(part_four, 1000, 1000, 2000))
    pass


if __name__ == '__main__':
    main()
