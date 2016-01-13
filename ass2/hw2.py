import random 

# question 1

random.seed(0)

def average(li):
    return sum(li) / len(li)

def flip():
    return random.randint(0,1)

def coin_result(num_flips):
    return [ flip() for i in range(num_flips) ]

def trial(sample_size, num_flips):
    return [ coin_result(num_flips) for i in range(sample_size) ]

def experiment(num_trials, sample_size, num_flips):
    return [ trial(sample_size, num_flips) for i in range(num_trials) ]

def v(trial, n):
    return average(trial[n])

def v1(trial):
    return v(trial, 0)

def v_rand(trial):
    return v(trial, random.randint(0, len(trial) - 1))

# def v_min(trial):
    # return v(trial, min([ (sum(trial[i]), i) for i in range(len(trial)) ])[1])

# def experiment_results(experiment):
#     return [ [v1(trial), v_rand(trial), v_min(trial)] for trial in experiment ] 

def experiment_results(experiment):
    return [ [v1(trial), v_rand(trial)] for trial in experiment ] 

def mean_experiment_results(experiment_results):
    return [ average(result) for result in zip(*experiment_results) ]

def question1():
    exp = experiment(100000, 1000, 10)
    # print(exp)
    exp_res = experiment_results(exp)
    # print(exp_res)
    # print(list(zip(*exp_res)))
    mean_exp_res = mean_experiment_results(exp_res)
    print(mean_exp_res)


# question 2


def main():
    question1()

if __name__ == '__main__':
    main()

