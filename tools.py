import numpy as np

from itertools import chain

# DATA CREATION

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

def random_set(n, target_function, transform=None):
    x = n_random_datapoint(n)
    y = get_y(target_function, x)
    if transform == None:
        return DataML((x, y))
    else:
        return DataML((x, y), transform)


class DataML:
    def __init__(self, data, transform=None):
        xy_as_tuple = type(data) == tuple or type(data) == list and len(data) == 2
        if xy_as_tuple :
            self.x = data[0]
            self.y = data[1]
        else:
            self.x = data[:,:data.shape[1]-1]
            self.y = data[:,data.shape[1]-1]
        if transform is None:
            self.z = self.x
        else:
            self.z = transform(self.x)
        if transform is None and not xy_as_tuple :
            self.z_y = data
        else:
            self.z_y = np.concatenate([self.z, np.array([self.y]).T], axis=1)

    def __repr__(self):
        z_repr = "input : z\n" + str(self.z)
        y_repr = "output : y\n" + str(self.y)
        return z_repr + "\n" + y_repr

def get_y(target_function, x):
    return np.apply_along_axis(target_function, 1, x)

# LINEAR FUNCTION CREATION 

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

def random_target_function():
    return create_linear_target_function(create_linear_function(*rand_line()))

# PERCEPTON LEARNING ALGORITHIM

def pla(x, y, weight=None, return_iterations=False):
    if weight is None:
        weight = linear_percepton(x,y)
    iterations = 0
    while True:
        mis_point_index = a_misclassified_point(x, y, weight)
        if mis_point_index is None:
            if return_iterations:
                return weight, iterations
            return weight
        weight = weight + x[mis_point_index] * y[mis_point_index]
        iterations += 1

def a_misclassified_point(x, y, weight):
    start = np.random.randint(0, len(x) - 1)
    for i in chain(range(start, len(x)), range(start)):
        if sign(np.dot(x[i], weight)) != y[i]:
            return i
    return None

# LINEAR PERCEPTON

def linear_percepton(x,y):
    xt_x = x.transpose().dot(x)
    xt_y = x.transpose().dot(y)
    inv_xt_x = np.linalg.inv(xt_x)
    return inv_xt_x.dot(xt_y)

# TESTING 

def weight_error(weight, z, y):
    learnt_output = classify(weight, z)
    return classified_error(learnt_output, y)

def classify(weights, z):
    vec_sign = np.vectorize(sign)
    return vec_sign(np.dot(z, weights))

def classified_error(learnt_output, real_output):
    equality_array = np.equal(learnt_output, real_output)
    return 1 - sum(equality_array) / len(equality_array)

def experiment(trial, trial_args, total_trials):
    results = [ trial(*trial_args) for _ in range(total_trials) ] 
    mean_results = np.mean(results, axis=0)
    return mean_results

# MISC

def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0

def output(simulations):
    print("conducting simulations...")
    sim = simulations()
    print("simulations done")
    for key in sorted(sim.keys()):
        print("""question""", key)
        print("".join([ "-" for i in range(len("question ") + len(str(key))) ]))
        print(sim[key][0], sim[key][1])
