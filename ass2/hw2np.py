import numpy as np

np.random.seed(0)

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


def main():
    exp = experiment(100000,1000,10)
    col_exp = collate_flip_results(exp)
    # print('exp\n', exp)
    # print('exp sum\n', col_exp)
    print('new avg1, avgrand', experiment_results(col_exp))

if __name__ == '__main__':
    main()
