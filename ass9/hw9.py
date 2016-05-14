import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir_path = os.path.dirname(file_dir)
sys.path.insert(0, tools_dir_path)

from tools import DataML, random_set, get_y, add_constant, second_order_nic, minimize_error_aug, svm, allexcept, a_vs_b, weight_error, classified_error, sign, output
import numpy as np

from tabulate import tabulate
from sklearn.svm import SVC
from sklearn.cluster import KMeans

np.random.seed(0)

# REGULARIZED LINEAR REGRESSION

def train_test(training_set, testing_set, learn, learn_args=[]):
    weight = learn(training_set.z, training_set.y, *learn_args)
    in_sample_error = weight_error(weight, training_set.z, training_set.y)
    out_of_sample_error = weight_error(weight, testing_set.z, testing_set.y)
    return [ in_sample_error, out_of_sample_error ]

# SUPPORT VECTOR MACHINES

def svm_que_helper():
    coordinates = [
            [1,0],
            [0,1],
            [0,-1],
            [-1,0],
            [0,2],
            [0,-2],
            [-2,0]
            ]
    result = [-1, -1, -1, 1, 1, 1, 1]
    def z1(x):
        return x[1]**2 - 2 * x[0] - 1
    def z2(x):
        return x[0]**2 - 2 * x[1] + 1
    # z = [ [z1(x), z2(x)] for x in coordinates ]
    # print(z)
    return DataML((coordinates, result))

def train_svc(training_set, svc):
    svc.fit(training_set.z, training_set.y)
    training_predicted = svc.predict(training_set.z)
    in_sample_error = classified_error(training_predicted, training_set.y)
    return len(svc.support_vectors_), in_sample_error

# RADIAL BASIS FUNCTIONS

def f(x):
    return sign(x[2] -x[1] + 1/4 * np.sin(np.pi * x[1]))

def generate_data():
    return random_set(100, f), random_set(100, f)

def svc_rbf_trial(training_set, testing_set):
    svc_rbf = SVC(kernel='rbf', C=10**6)
    svc_rbf.fit(training_set.z, training_set.y)
    svc_ein, svc_eout = [
            classified_error(svc_rbf.predict(t_set.z), t_set.y)
            for t_set in [training_set, testing_set] ]
    print(svc_ein)
    if svc_ein > 10**-6:
        return None
    return svc_eout

def get_cluster_centers(z, k):
    return KMeans(n_clusters=k).fit(z).cluster_centers_

def gaussian(gamma, x, center):
    # print(weight, center)
    return np.exp(-1 * gamma * np.linalg.norm(x - center))

def gen_fixed_gaussian(gamma, center):
    def fixed_guassian(x):
        return gaussian(gamma, x, center)
    return fixed_guassian

def get_phi(z, gammas, centers):
    assert len(gammas) == len(centers)
    fixed_guassians = [ gen_fixed_gaussian(gammas[i], centers[i])
        for i in range(len(gammas)) ]
    return np.vstack([ 
        np.apply_along_axis(fixed_guassian, 1, z)
        for fixed_guassian in fixed_guassians]).transpose()

def reg_rbg_learn(z, y, k):
    cluster_centers = get_cluster_centers(z, k)
    initial_gammas = np.ones(k)
    initial_phi = get_phi(z, initial_gammas, cluster_centers) # gammas initally set to one
    initial_weight = lmsq(initial_phi, y)
    gamma_weight = expectation_maximization(
            z, y, 
            cluster_centers, 
            initial_gammas,
            initial_weight)
    return gamma_weight

def expectation_maximization(
        z, y, 
        centers, 
        initial_gammas, 
        initial_weight):
    old_error = total_rbf_lmsq_error(z, y, initial_gammas, initial_weight, centers)
    new_gammas, iterations = stochastic_gradient_descent(z, y, 
            rbf_lmsq_error_d_wrt_ith_g,
            initial_gammas,
            {'w' : initial_weight, 'u' : centers})
    new_weight = get_weight(z, y, new_gammas, centers)
    print(new_gammas, iterations)
    new_error = total_rbf_lmsq_error(z, y, new_gammas, new_weight, centers)
    assert old_error >= new_error
    iterations = 1
    while old_error - new_error > 0.01:
        iterations += 1
        print(iterations)
        old_error = new_error
        new_gammas = stochastic_gradient_descent(z, y, 
                rbf_lmsq_error_d_wrt_ith_g,
                new_gammas,
                {'w' : new_weight, 'u' : centers})[0]
        new_weight = get_weight(z, y, new_gammas, centers)
        new_error = total_rbf_lmsq_error(z, y, new_gammas, new_weight, centers)
    return new_gammas, new_weight
        



def stochastic_gradient_descent(z, y, derivative, initial_alphas, kwargs=dict()):
    """optimizing for alphas"""
    def gen_ith_derivatives(derivative, i, kwargs):
        def ith_derivative(x, y, alphas):
            return derivative(x, y, alphas, i=i, **kwargs)
        return ith_derivative
    gradient = [ gen_ith_derivatives(derivative, i, kwargs) 
            for i in range(len(initial_alphas)) ]
    old_run_alphas = epoch(z, y, initial_alphas, gradient)
    new_run_alphas = epoch(z, y, old_run_alphas, gradient)
    i = 0
    while np.linalg.norm(old_run_alphas - new_run_alphas) > 0.01:
        i += 1 
        old_run_alphas = new_run_alphas
        new_run_alphas = epoch(z, y, new_run_alphas, gradient)
    return new_run_alphas, i

def epoch(z, y, alphas, gradient):
    LEARNING_RATE = 0.01
    data_index_iter = np.random.permutation(len(z))
    for i in data_index_iter:
        alphas = alphas - LEARNING_RATE * np.array(
                [ derivative(z[i], y[i], alphas) for derivative in gradient ])
    return alphas

def get_weight(z, y, gammas, centers):
    return lmsq(get_phi(z, gammas, centers), y)

def lmsq(phi, y):
    return np.linalg.pinv(phi).dot(y)

def rbf_h(x, g, w, u):
    return sum([ w[i] * gaussian(g[i], x, u[i])
                for i in range(len(g)) ])

def total_rbf_lmsq_error(z, y, gammas, weight, centers):
    phi = get_phi(z, gammas, centers)
    return np.linalg.norm(phi.dot(weight) - y)

def rbf_lmsq_error(x, y, gammas, weight, centers):
    """least mean square error"""
    assert len(gammas) == len(weights) and len(gammas) == len(centers)
    return np.linalg.norm(rbf_h(x, gammas, weight, centers) - y)

def rbf_lmsq_error_d_wrt_ith_g(x, y, g, w, u, i):
    """
    radial basis function 
    least mean square error derivative 
    with respect to the ith gamma
    g : gammas
    w : weights
    u : centers
    """
    return -2 * w[i] * np.linalg.norm(x - u[i]) * gaussian(g[i], x, u[i]) * (rbf_h(x, g, w, u) - y)

def reg_rbg_trial(training_set, testing_set, k=9):
    gamma_weight = reg_rbg_learn(training_set.z, training_set.y, k)

def trial():
    # import matplotlib.pyplot as plt
    # space = np.linspace(-1,1,25)
    # print(space)
    # x = np.array([ [ 1, a, b] for a in space for b in space ])
    # print(x)
    # real_y = get_y(f, x)
    # real_y_pos_i = real_y == 1
    # x_pos = x[real_y_pos_i]
    # # plt.plot(x_pos[:,1], x_pos[:,2], 'bo')
    # x_neg = x[real_y == -1]
    # # plt.plot(x_neg[:,1], x_neg[:,2], 'ro')
    training_set, testing_set = generate_data()
    # train_y_pos_i = training_set.y == 1
    # train_x_pos = training_set.z[train_y_pos_i]
    # plt.plot(train_x_pos[:,1], train_x_pos[:,2], 'go')
    # train_x_neg = training_set.z[training_set.y == -1]
    # plt.plot(train_x_neg[:,1], train_x_neg[:,2], 'o', c='orange')
    results = reg_rbg_trial(training_set, testing_set)
    # plt.plot(cluster_centers[:,1], cluster_centers[:,2], 'b^')
    # plt.show()
    # print(svc_rbf_trial(training_set, testing_set))

def main():
    note = \
    """
    important note, you might be suprised that in some of the problems,
    going to a higher dimensional feature space results in a greater 
    in sample error, something thats not suppose to happen seeing that 
    higher dimensional feature spaces are suppose to be a superset of 
    the original feature space. This paradoxical result can be 
    explained by the learning algorithim optimizing for linear
    result instead of classification
    """
    print(note)
    # output(simulations)
    trial()

def simulations():
    que = {}
    training_data = np.genfromtxt(os.path.join(file_dir, "features.train"))
    testing_data = np.genfromtxt(os.path.join(file_dir, "features.test"))
    def convert_raw(t_data):
        return DataML((t_data[:,1:], np.array(t_data[:,0], dtype="int")))
    initial_training_set = convert_raw(training_data)
    initial_testing_set = convert_raw(testing_data)
    def transform_help(transform, *col_data_sets):
        return [ DataML((transform(data_set.z), data_set.y)) 
                for data_set in col_data_sets ]
    constant_training_set, constant_testing_set = transform_help(
            add_constant, initial_training_set, initial_testing_set)
    allexcept_constant_train_test_li = [
            allexcept(digit, constant_training_set, constant_testing_set)
            for digit in range(10) ]
    no_transform_errors = [ train_test(
        *train_test_sets, 
        minimize_error_aug,
        [1])
        for train_test_sets in allexcept_constant_train_test_li ]
    in_sample_error_5_9 = [ 
            error_list[0] for error_list in no_transform_errors[5:10] ]
    min_arg = np.argmin(in_sample_error_5_9) + 5
    min_error = min(in_sample_error_5_9)
    que[7] = ("digit with lowest in sample error : ", 
            str(min_arg) + ", " + str(min_error))

    second_order_training_set, second_order_testing_set = transform_help(
            second_order_nic, initial_training_set, initial_testing_set)
    allexcept_second_order_train_test_li = [
            allexcept(
                digit, 
                second_order_training_set, 
                second_order_testing_set)
            for digit in range(10) ]
    transform_errors = [ train_test(
        *train_test_sets, 
        minimize_error_aug,
        [1])
        for train_test_sets in allexcept_second_order_train_test_li ]
    out_of_sample_error_0_4 = [ 
            error_list[1] for error_list in transform_errors[:5] ]
    min_arg = np.argmin(out_of_sample_error_0_4)
    min_error = min(out_of_sample_error_0_4)
    que[8] = ("digit with lowest out of sample error : ", 
            str(min_arg) + ", " + str(min_error))

    tables = [ [
        ["no transform"] + no_transform_errors[i], 
        ["transform"] + transform_errors[i] ]
        for i in range(10) ]

    pretty_tables = [ tabulate(
        table, 
        headers=["","in sample", "out of sample"])
        for table in tables ] 

    tables_string = "\n".join(
            ["\ndigit {}\n".format(i) + str(pretty_tables[i])
            for i in range(len(pretty_tables)) ]
            )

    que[9] = ("effectiveness of feature transform on 0 and 9",
            tables_string
            )
    one_v_five_second_order_sets = a_vs_b(
            1, 5,
           second_order_training_set, 
           second_order_testing_set)
    errors_lambda = [ train_test(
        *one_v_five_second_order_sets,
        minimize_error_aug,
        [alpha]) 
        for alpha in [0.01, 1] ]
    pretty_table = tabulate( 
            [ ["lambda 0.01"] + errors_lambda[0],
              ["lambda 1"] + errors_lambda[1] ],
            headers=["", "in sample", "out of sample"]
            )
    que[10] = ("errors from changing lambda for 1 vs 5\n", "\n" + str(pretty_table) + "\n\nevidence of overfitting as increased constraint improves performance")
    total_support_vectors, in_sample_error = train_svc(
            transform_help(add_constant, svm_que_helper())[0],
            SVC(kernel="poly", degree=2, C=float("infinity")))
    que[12] = ("total support vectors :", total_support_vectors)
    return que

ans = {
        1 : 'e', # The dimensionality of the Z space should be 65
        2 : 'e', 
        3 : 'a',
        4 : 'd',
        5 : 'a',
        6 : 'b',
        7 : 'a',
        8 : 'b',
        9 : '', #all the statements are wrong although d comes close 
        10 : 'a',
        11 : 'c',
        12 : 'c',
        13 : '',
        14 : '',
        15 : '',
        16 : '',
        17 : '',
        18 : '',
        19 : '',
        20 : '',
        }

if __name__ == "__main__":
    main()
