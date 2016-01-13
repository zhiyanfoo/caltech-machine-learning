import numpy as np
from itertools import chain

np.random.seed(0)

def n_random_datapoint(n):
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

def classify_point(data_point, target_function):
    if data_point[2] - target_function(data_point[1]) < 0:
        return -1
    else:
        return 1

def classify_data(raw_data, target_function):
    return np.array([ [data_point, classify_point(data_point, target_function)] for data_point in raw_data ])

def binary_percepton(data):
    weight_vectors = np.zeros(len(data[0,0]))
    weight, iterations = update(weight_vectors, data, 0)
    return weight, iterations

def update(weight, data, iterations):
    mis_point = a_misclassified_point(data, weight)
    if mis_point is None:
        return weight, iterations
    new_weight = weight + mis_point[1] * mis_point[0]
    return update(new_weight, data, iterations + 1)

def a_misclassified_point(data, weight):
    start = np.random.randint(0, len(data) - 1)
    for i in chain(range(0, start), range(start,len(data))):
        if sign(data[i,0], weight) != data[i,1]:
            return data[i]
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
    x = data[:,0]
    print("x")
    print(x)
    y = np.matrix(data[:,1])
    # print("y")
    # print(y)
    xt_x = x.transpose() * x
    print(xt_x)
    # xt_y = x.transpose() * y
    # print(xt_y)
    # inv_xt_x = np.linalg.inv(xt_x)
    # return np.dot(np.linalg.inv(xt_x), np.dot(x.transpose(), y))

def check_classification(data, lin_target_function):
    import matplotlib.pyplot as plt
    plt.plot([-1, 1], [lin_target_function(-1), lin_target_function(1)], '-')
    plt.plot([-1,1],[lin_target_function(-1),lin_target_function(1)], '-')
    xy_plus = [ [data_point[0][1], data_point[0][2]] for data_point in data if data_point[1] == 1]
    xy_minus = [ [data_point[0][1], data_point[0][2]] for data_point in data if data_point[1] == -1]
    p1 = zip(*xy_plus)
    p2 = zip(*xy_minus)

    x1, y1 = p1
    x2, y2 = p2

    plt.plot(x1, y1, 'ro')
    plt.plot(x2, y2, 'go')
    plt.axis([-1, 1, -1, 1])
    plt.show()

def main():
    raw_data = n_random_datapoint(8)
    # print(raw_data)
    m, c = rand_line()
    # print(m ,c)
    target_function = create_lin_target_function(m, c)
    data = classify_data(raw_data, target_function)
    # print(data)
    # print(binary_percepton(data))
    print(linear_percepton(data))
    # check_classification(data, target_function)

if __name__ == "__main__":
    main()
