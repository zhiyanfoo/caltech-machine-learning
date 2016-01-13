import sys
import random
from itertools import chain
from pprint import pprint


random.seed(0)
sys.setrecursionlimit(2000)

def rand_float():
    return random.uniform(-1,1)

def data_point():
    return [ rand_float() ,rand_float() ] 

def dot_product(x,y):
    if len(x) != len(y):
        raise ValueError("For dotproduct(seq1, seq2), length of seq1 must equal length of seq2")
    return sum( x[i] * y[i] for i in range(len(x)) )  

def sign(x,w):
    x_dot_w = dot_product(x,w)
    if x_dot_w < 0:
        return -1
    elif x_dot_w > 0:
        return 1
    else:
        return 0

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


def n_random_datapoint(n):
    return [ [1] +  data_point() for i in range(n) ]

def classify(data_point, target_function):
    if data_point[2] - target_function(data_point[1]) < 0:
        return -1
    else:
        return 1

def data_classified(target_function, raw_data):
    ''' [ [ x0 ... xn], y ] ... ]
    [[[1, 0.02254944273721704, -0.19013172509917142], 1],
     [[1, 0.5675971780695452, -0.3933745478421451], -1],
     [[1, -0.04680609169528838, 0.1667640789100624], 1]]
    '''
    return [ [data_point, classify(data_point, target_function)] for data_point in raw_data ] 

def a_misclassified_point(data, weight):
    start = random.randint(0, len(data) - 1)
    for i in chain(range(0, start), range(start,len(data))):
        if sign(data[i][0], weight) != data[i][1]:
            return data[i]
    return None
        
def check_classification(data, lin_target_function):
    import matplotlib.pyplot as plt
    plt.plot([-1,1],[lin_target_function(-1),lin_target_function(1)], '-')
    xy_plus = [ [data_point[0][1], data_point[0][2]] for data_point in data if data_point[1] == 1]
    xy_minus = [ [data_point[0][1], data_point[0][2]] for data_point in data if data_point[1] == -1]
    p1 = zip(*xy_plus)
    p2 = zip(*xy_minus)

    x1, y1 = p1
    x2, y2 = p2


    plt.plot(x1, y1, 'ro')
    plt.plot(x2, y2, 'go')
    plt.show()
    
def misclassified_points(data, weight):
    return [ data_point for data_point in data if sign(data_point[0], weight) != data_point[1] ]

def perception_learning_lin(data):
    weight_vectors = [ 0 for i in range(len(data[0][0]))]
    weight, iterations = update(weight_vectors, data, 0)
    return weight, iterations

def update(weight, data, iterations):
    mis_point = a_misclassified_point(data, weight)
    if mis_point == None:
        return weight, iterations
    new_weight = [ weight[i] + mis_point[1] *  mis_point[0][i] for i in range(len(weight)) ]
    return update(new_weight, data, iterations + 1)

def main():
    lin_target_function = create_lin_target_function(*rand_line())
    data = data_classified(lin_target_function, n_random_datapoint(1000))

    weight, iterations = perception_learning_lin(data)
    print(iterations)

    pprint(len(misclassified_points(data, weight)))

if __name__ == "__main__":
    main()
