import numpy as np

from itertools import chain

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

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
    return add_constant(np.random.uniform(-1,1, size=(n,2)))

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
        if xy_as_tuple:
            self.x = np.copy(data[0])
            self.y = np.copy(data[1])
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

# TRANSFORMS

def transform(x):
    """
    transform             
    x1 x2  --->   1 x1 x2 x1**2 x2**2 x1x2 |x1 - x2| |x1 + x2|
    """
    ones = np.ones(len(x))
    x1 = x[:,0]
    x2 = x[:,1]
    x1_sqr = x1**2
    x2_sqr = x2**2
    x1x2 = x1 * x2
    abs_x1_minus_x2 = abs(x1-x2)
    abs_x1_plus_x2 = abs(x1+x2)
    return np.stack([ones, x1, x2, x1_sqr, x2_sqr, x1x2, abs_x1_minus_x2, abs_x1_plus_x2], axis=1)

def add_constant(x):
    """
    transform             
    x1 x2  --->   1 x1 x2
    """
    return np.insert(x,0,1, axis=1)

def second_order(x):
    """
    transform             
    1 x1 x2  --->   1 x1 x2 x1x2 x1**2 x2**2
    """
    ones = x[:, 0]
    x1 = x[:, 1]
    x2 = x[:, 2]
    x1_sqr = x1**2
    x2_sqr = x2**2
    x1x2 = x1 * x2
    return np.stack([ones, x1, x2, x1x2, x1_sqr, x2_sqr], axis=1)

def second_order_nic(x):
    """
    transform             
    x1 x2  --->   1 x1 x2 x1x2 x1**2 x2**2

    nic : no initial constant
    """
    ones = np.ones(len(x))
    x1 = x[:, 0]
    x2 = x[:, 1]
    x1_sqr = x1**2
    x2_sqr = x2**2
    x1x2 = x1 * x2
    return np.stack([ones, x1, x2, x1x2, x1_sqr, x2_sqr], axis=1)

# PERCEPTON LEARNING ALGORITHIM

def pla(x, y, weight=None, return_iterations=False):
    """
    Percepton Learning Algorithim (PLA)
    Returns: weights
    Caveat: only works for linearly seperable data, if not will run for infinity
    """
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

# CONSTRAINED LEARNING

def minimize_error_aug(z,y,a):
    """
    minimize
    d_Ein = Z(Z*w - y) + a*w = 0
    (Z*Z + a*I)^-1 * Z*y) = w
    Returns: weights
    """
    zz = z.transpose().dot(z)
    zz_plus_ai = zz + a * np.identity(len(zz))
    inv_zz_plus_ai = np.linalg.inv(zz_plus_ai)
    zy = z.transpose().dot(y)
    inv_zz_plus_ai_zy = inv_zz_plus_ai.dot(zy)
    return inv_zz_plus_ai_zy 

# SUPPORT VECTOR MACHINE

def svm(x, y):
    """
    classification SVM

    Minimize
    1/2 * w^T w
    subject to
    y_n (w^T x_n + b) >= 1
    """
    weights_total = len(x[0])
    I_n = np.identity(weights_total-1)
    P_int =  np.vstack(([np.zeros(weights_total-1)], I_n))
    zeros = np.array([np.zeros(weights_total)]).T
    P = np.hstack((zeros, P_int))
    q = np.zeros(weights_total)
    G = -1 * vec_to_dia(y).dot(x)
    h = -1 * np.ones(len(y))
    matrix_arg = [ matrix(x) for x in [P,q,G,h] ]
    sol = solvers.qp(*matrix_arg)
    return np.array(sol['x']).flatten()

def vec_to_dia(y):
    dia = [ [ 0 for i in range(i) ] 
            + [y[i]] 
            + [ 0 for i in range(i,len(y) - 1) ]  
            for i in range(len(y)) ]
    return np.array(dia, dtype='d')

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

# MULTI CLASSIFICATION TO BINARY CLASSIFICATION

def allexcept(digit, *col_data_sets):
    """
    col_data_set : collection of data sets
    Returns : list of datasets subject to the following transformation
    if a in y != digit
        a = -1
    """
    copies_of_data_sets = [ DataML((data_set.z, data_set.y)) for data_set in col_data_sets ]
    for data_set in copies_of_data_sets:
        truth_table = data_set.y != digit 
        data_set.y[truth_table] = -1
    return copies_of_data_sets

def a_vs_b(a, b, *col_data_sets):
    """
    col_data_set : collection of data sets
    Returns : list of datasets subject to the following transformation
    filter x in y if != a or b
    set y values to either 1 or 0 
    depending on whether the original was a or b originally
    """
    copies_of_data_sets = [ DataML((data_set.z, data_set.y)) for data_set in col_data_sets ]
    for i in range(len(copies_of_data_sets)):
        data_set = copies_of_data_sets[i]
        truth_a = data_set.y == a
        truth_b = data_set.y == b
        ones = np.ones(sum(truth_a))
        neg_ones = -1 * np.ones(sum(truth_b))
        z = np.vstack([data_set.z[truth_a], data_set.z[truth_b]])
        y = np.concatenate([ones, neg_ones])
        copies_of_data_sets[i] = DataML((z,y))
    return copies_of_data_sets

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

