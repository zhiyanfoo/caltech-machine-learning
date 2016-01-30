import numpy as np
from itertools import chain

np.random.seed(0)

def n_random_datapoint(n):
    '''
    [[ 1.          0.09762701  0.43037873]
     [ 1.          0.20552675  0.08976637]
     [ 1.         -0.1526904   0.29178823]
     [ 1.         -0.12482558  0.783546  ]
     [ 1.          0.92732552 -0.23311696]
     [ 1.          0.58345008  0.05778984]
     [ 1.          0.13608912  0.85119328]
     [ 1.         -0.85792788 -0.8257414 ]]
    '''
    return np.insert(np.random.uniform(-1,1, size=(n,2)), 0, 1, axis=1)

def rand_line():
    x1, y1, x2, y2 = np.random.uniform(-1,1, size=4)
    m = (y1 - y2) / (x1 - x2)
    c = y1 - x1 * m
    return m, c

def create_linear_function(m,c):
    def linear_function(x):
        return x*m + c
    return linear_function

def create_linear_target_function(linear_function):
    def linear_target_function(data_point):
        if data_point[2] - linear_function(data_point[1]) < 0:
            return -1
        else:
            return 1
    return linear_target_function

def non_linear_target_function(data_point):
    return sign(data_point[1]**2 + data_point[2]**2 - 0.6)


def classify_data(raw_data, target_function):
    classified = np.array([ target_function(data_point) for data_point in raw_data ])
    return {'raw' : raw_data, 'classified' : classified}

def classify_data_linear_binary_random(raw_data):
    m, c = rand_line()
    linear_function = create_linear_function(m, c)
    linear_target_function = create_linear_target_function(linear_function)
    data = classify_data(raw_data, linear_target_function)
    return data, linear_target_function

def binary_percepton(data):
    # weight_vectors = np.zeros(len(data['raw'][0]))
    weight_vectors = linear_percepton(data)
    weight, iterations = update(weight_vectors, data, 0)
    return weight, iterations

def binary_percepton(data, weight_vectors=None):
    # weight_vectors = np.zeros(len(data['raw'][0]))
    if weight_vectors != None:
        weight_vectors = linear_percepton(data)
    weight, iterations = update(weight_vectors, data, 0)
    return weight, iterations

def update(weight, data, iterations):
    mis_index = a_misclassified_point(data, weight)
    if mis_index is None:
        return weight, iterations
    new_weight = weight + data['raw'][mis_index] * data['classified'][mis_index]
    return update(new_weight, data, iterations + 1)

def a_misclassified_point(data, weight):
    start = np.random.randint(0, len(data['raw']) - 1)
    for i in chain(range(0, start), range(start, len(data['raw']))):
        if sign(np.dot(data['raw'][i], weight)) != data['classified'][i]:
            return i
    return None

def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0

def linear_percepton(data):
    x = data['raw']
    y = data['classified']
    xt_x = x.transpose().dot(x)
    xt_y = x.transpose().dot(y)
    inv_xt_x = np.linalg.inv(xt_x)
    return inv_xt_x.dot(xt_y)

def trial(in_sample, out_sample):
    raw_data = n_random_datapoint(out_sample)
    data, target_function = classify_data_linear_binary_random(raw_data)
    training_indices = np.random.choice(out_sample, size=in_sample, replace=False)
    training_raw = data['raw'][training_indices, :]
    training_classified = data['classified'][training_indices]
    training_data = { 'raw' : training_raw, 'classified' : training_classified } 
    linear_weight = linear_percepton(training_data)
    return check_error(training_data, linear_weight), check_error(data, linear_weight) 

def check_error(data, linear_weight):
    linear_classification = [ sign(np.dot(x, linear_weight)) for x in data['raw'] ]
    n_misclassified_points = len(data['classified']) - sum(linear_classification == data['classified'])
    return n_misclassified_points / len(data['raw'])

def average_trial_results(num_trials, in_sample, out_sample):
    trials = np.array([ trial(100, 1000) for _ in range(num_trials) ])
    return np.mean(trials, axis=0)


def main():
    # trial_results = trial(100, 1000)
    # print(trial_results)
    # print(average_trial_results(1000, 100, 1000))
    pass



def check_classification(data, linear_function, weight=None):
    import matplotlib.pyplot as plt
    
    if weight != None:
        def color(i):
            return sign(np.dot(data['raw'][i], weight))
    else:
        def color(i):
            return data['classified'][i]


    plt.plot([-1, 1], [linear_function(-1), linear_function(1)], '-')
    plt.plot([-1,1],[linear_function(-1),linear_function(1)], '-')
    xy_plus = [ [data['raw'][i,1], data['raw'][i,2]] for i in range(len(data['raw'])) if color(i) == 1]
    # print('xy_plus')
    # print(xy_plus)
    # xy_minus = [ [data['raw'][i,1], data['raw'][i,2]] for i in range(len(data['raw'])) if data['classified'][i]  == -1]
    xy_minus = [ [data['raw'][i,1], data['raw'][i,2]] for i in range(len(data['raw'])) if color(i)  == -1]
    # print('xy_minus')
    # print(xy_minus)
    p1 = zip(*xy_plus)
    p2 = zip(*xy_minus)

    x1, y1 = p1
    x2, y2 = p2

    plt.plot(x1, y1, 'ro')
    plt.plot(x2, y2, 'go')
    plt.axis([-1, 1, -1, 1])
    plt.show()

if __name__ == "__main__":
    main()

