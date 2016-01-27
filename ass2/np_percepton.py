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

def create_lin_target_function(m,c):
    def lin_target_function(x):
        return x*m + c
    return lin_target_function


def classify_data(raw_data, target_function):
    classified = np.array([ classify_point(data_point, target_function) for data_point in raw_data ])
    return {'raw' : raw_data, 'classified' : classified}

def classify_data_linear_binary_random(raw_data):
    m, c = rand_line()
    target_function = create_lin_target_function(m, c)
    data = classify_data(raw_data, target_function)
    return data, target_function

def classify_point(data_point, target_function):
    if data_point[2] - target_function(data_point[1]) < 0:
        return -1
    else:
        return 1

def binary_percepton(data):
    weight_vectors = np.zeros(len(data['raw'][0]))
    weight, iterations = update(weight_vectors, data, 0)
    return weight, iterations

def update(weight, data, iterations):
    mis_index = a_misclassified_point(data, weight)
    if mis_index is None:
        return weight, iterations
    new_weight = weight + data['raw'][mis_index] * data['classified'][mis_index]
    # print(new_weight)
    return update(new_weight, data, iterations + 1)

def a_misclassified_point(data, weight):
    start = np.random.randint(0, len(data['raw']) - 1)
    for i in chain(range(0, start), range(start, len(data['raw']))):
        if sign(data['raw'][i], weight) != data['classified'][i]:
            return i
    return None

def sign(x,w):
    x_dot_w = np.dot(x,w)
    if x_dot_w < 0:
        return -1
    elif x_dot_w > 0:
        return 1
    else:
        return 0

def linear_percepton(data):
    x = data['raw']
    y = data['classified']
    print("x")
    print(x)
    print("y")
    print(y)
    xt_x = x.transpose().dot(x)
    print('xt_x')
    print(xt_x)
    xt_y = x.transpose().dot(y)
    print('xt_y')
    print(xt_y)
    inv_xt_x = np.linalg.inv(xt_x)
    print('inv_xt_x')
    print(inv_xt_x)
    return inv_xt_x.dot(xt_y)

def check_classification(data, lin_target_function, weight=None):
    import matplotlib.pyplot as plt
    
    if weight != None:
        def color(i):
            return sign(data['raw'][i], weight)
    else:
        def color(i):
            return data['classified'][i]


    plt.plot([-1, 1], [lin_target_function(-1), lin_target_function(1)], '-')
    plt.plot([-1,1],[lin_target_function(-1),lin_target_function(1)], '-')
    # xy_plus = [ [data['raw'][i,1], data['raw'][i,2]] for i in range(len(data['raw'])) if data['classified'][i] == 1]
    xy_plus = [ [data['raw'][i,1], data['raw'][i,2]] for i in range(len(data['raw'])) if color(i) == 1]
    print('xy_plus')
    print(xy_plus)
    # xy_minus = [ [data['raw'][i,1], data['raw'][i,2]] for i in range(len(data['raw'])) if data['classified'][i]  == -1]
    xy_minus = [ [data['raw'][i,1], data['raw'][i,2]] for i in range(len(data['raw'])) if color(i)  == -1]
    print('xy_minus')
    print(xy_minus)
    p1 = zip(*xy_plus)
    p2 = zip(*xy_minus)

    x1, y1 = p1
    x2, y2 = p2

    plt.plot(x1, y1, 'ro')
    plt.plot(x2, y2, 'go')
    plt.axis([-1, 1, -1, 1])
    plt.show()

def trial(in_sample, out_sample):
    raw_data = n_random_datapoint(out_sample)
    training_indices = np.random.choice(out_sample, size=in_sample, replace=False)
    print('training_indices')
    print(training_indices)
    training_data = raw_data[training_indices,:]
    print('training_data')
    print(training_data)
    data, target_function = classify_data_linear_binary_random(raw_data)
    linear_weight = linear_percepton(data)
    return check_error(training_data), return check_error(raw_data)



def check_error(data, linear_weight):
    linear_classification = [ sign(x, linear_weight) for x in data['raw'] ]
    n_misclassified_points = len(data['classified']) - sum(linear_classification == data['classified'])
    check_classification(data, target_function, linear_weight)
    print(n_misclassified_points)
    return n_misclassified_points / len(data['raw'])


def main():
    # raw_data = n_random_datapoint(140)
    # print('raw_data')
    # print(raw_data)
    # m, c = rand_line()
    # print(m ,c)
    # target_function = create_lin_target_function(m, c)
    # data = classify_data(raw_data, target_function)
    # print('data')
    # print(data)
    # print('binary_percepton')
    # print(binary_percepton(data))
    # print('linear_percepton')
    # lin_per = linear_percepton(data)
    # print(lin_per)
    # check_classification(data, target_function, lin_per)
    trial(100, 1000)


if __name__ == "__main__":
    main()
