import random
import matplotlib.pyplot as plt

random.seed(0)

def rand_float():
    return random.uniform(-1,1)

def data_point():
    return [ rand_float() ,rand_float() ] 

def dot_product(x,y):
    return sum([ x[i] * y[i] for i in range(len(x)) ]) 

def sign(x,w):
    x_dot_w = dot_product(x,w)
    if x_dot_w < 0:
        return -1
    elif x_dot_w == 0:
        return 0
    else:
        return 1

def rand_line():
    x1, y1 = data_point()
    x2, y2 = data_point()
    m = (y1 - y2) / (x1 - x2)
    c = y1 - x1 * m
    return m, c


def create_lin_target_function(m,c):
    def lin_target_function(x):
        return x*m + c
    return lin_target_function


def n_input_data(n):
    return [ [1] +  data_point() for i in range(n) ]

def data_classified(lin_target_function, raw_data):
    data = list()
    for data_point in raw_data:
        diff = data_point[2] - lin_target_function(data_point[1])
        if diff < 0:
            truncated_val = -1
        else:
            truncated_val = 1
        data.append([data_point, truncated_val])
    return data

def perception_learning(data, lin_target_function):
    weight_vectors = [ 0 for i in range(3)]
    misclassified_points = data[:]
    weight, iterations = update(weight_vectors, misclassified_points, 0)
    print("weight:", weight, "iterations:", iterations)
    perception_learning_checker(data, weight, lin_target_function)

def misclassified_points(data, weight):
    return [ data_point for data_point in data if sign(data_point, weight) != data_point[1] ]

def first_misclassified_point(data, weight):
    # None if no missclassified point
    random.randint(0,len
    for data_point in data:
        if sign(data_point, weight) != data_point[1]:
            return data_point
    return None
    

def update(weight, data, misclassified_points, iterations):
    if len(misclassified_points) == 0:
        return weight, iterations
    point = misclassified_points[0]
    new_weight = [ weight[i] + point[1] * point[0][i]  for i in range(len(point[0])) ]
    for i in range(len(misclassified_points) - 1, -1, -1):
        corrected = sign(misclassified_points[i][0], new_weight) == misclassified_points[i][1]
        if corrected:
            misclassified_points.pop(i)
    return update(new_weight, misclassified_points, iterations + 1)

lin_target_function = create_lin_target_function(*rand_line())
data = data_classified(lin_target_function, n_input_data(10))
print(data)

perception_learning(data, lin_target_function)
