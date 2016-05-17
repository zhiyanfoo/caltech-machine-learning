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
    svc_rbf = SVC(kernel='rbf', C=10**4)
    svc_rbf.fit(training_set.z, training_set.y)
    svc_ein, svc_eout = [
            classified_error(svc_rbf.predict(t_set.z), t_set.y)
            for t_set in [training_set, testing_set] ]
    if svc_ein > 10**-6:
        return None
    return svc_eout

def reg_rbg_trial(training_set, testing_set, k, gammas):
    gammas, weight, centers = reg_rbg_learn(
            training_set.z, training_set.y, k, gammas)
    in_sample_error, out_of_sample_error = [ 
            classified_error(
                rbf_predict(t_set.z, gammas, weight, centers), 
                t_set.y)
            for t_set in [training_set, testing_set] ]
    return in_sample_error, out_of_sample_error

def rbf_predict(z, gammas, weight, centers):
    vec_sign = np.vectorize(sign)
    phi = get_phi(z, gammas, centers)
    return vec_sign(phi.dot(weight))

def get_cluster_centers(z, k):
    kmeans = KMeans(n_clusters=k).fit(z) 
    return kmeans.cluster_centers_

def gaussian(gamma, x, center):
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

def reg_rbg_learn(z, y, k, gammas):
    centers = get_cluster_centers(z, k)
    phi = get_phi(z, gammas, centers)
    weight = lmsq(phi, y)
    return gammas, weight, centers

def expectation_maximization(
        z, y, 
        centers, 
        initial_gammas, 
        initial_weight):
    """
    optimize for gammas-using stochastic_gradient_descent-in addition
    to weights, not used in the problem set"""
    # from tools import stochastic_gradient descent 
    old_error = total_rbf_lmsq_error(z, y, initial_gammas, initial_weight, centers)
    new_gammas, iterations = stochastic_gradient_descent(z, y, 
            rbf_lmsq_error_d_wrt_ith_g,
            initial_gammas,
            {'w' : initial_weight, 'u' : centers})
    new_weight = get_weight(z, y, new_gammas, centers)
    new_error = total_rbf_lmsq_error(z, y, new_gammas, new_weight, centers)
    iterations = 1
    while abs(old_error - new_error) > 0.01:
        iterations += 1
        old_error = new_error
        new_gammas = stochastic_gradient_descent(z, y, 
                rbf_lmsq_error_d_wrt_ith_g,
                new_gammas,
                {'w' : new_weight, 'u' : centers})[0]
        new_weight = get_weight(z, y, new_gammas, centers)
        new_error = total_rbf_lmsq_error(z, y, new_gammas, new_weight, centers)
    return new_gammas, new_weight

def get_weight(z, y, gammas, centers):
    return lmsq(get_phi(z, gammas, centers), y)

def lmsq(phi, y):
    return np.linalg.pinv(phi).dot(y)

def rbf_h(x, g, w, u):
    return sum([ w[i] * gaussian(g[i], x, u[i])
                for i in range(len(g)) ])

def total_rbf_lmsq_error(z, y, gammas, weight, centers):
    """
    total radial bassis function least mean square error
    g : gammas
    w : weights
    u : centers
    """
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


def trial(total_trials, k, gammas):
    svc_eout_li = list()
    reg_ein_li = list()
    reg_eout_li = list()
    total_hard_margin_svc_failure = 0
    while len(svc_eout_li) < total_trials:
        training_set, testing_set = generate_data()
        svc_eout = svc_rbf_trial(training_set, testing_set)
        if svc_eout == None:
            total_hard_margin_svc_failure += 1
            continue
        svc_eout_li.append(svc_eout)
        reg_ein, reg_eout = reg_rbg_trial(training_set, testing_set, k, gammas)
        reg_ein_li.append(reg_ein)
        reg_eout_li.append(reg_eout)
    return total_hard_margin_svc_failure, np.array(svc_eout_li), np.array(reg_ein_li), np.array(reg_eout_li)


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
    output(simulations)

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
    total_trials = 30
    class SVC_REGULAR:
        def __init__(self, total_trials, k, gammas):
            self.total_hard_margin_svc_failure, \
            self.svc_eout_li, \
            self.reg_ein_li, \
            self.reg_eout_li = trial(total_trials, k, gammas)

    k9_g1x5 = SVC_REGULAR(total_trials, 9, 1.5 * np.ones(9))
    que[13] = ("total hard margin svc failure percentage :", k9_g1x5.total_hard_margin_svc_failure / total_trials)
    que[14] = ("svc rbf better than regular rbf percentage (k=9):", 
            sum(k9_g1x5.svc_eout_li < k9_g1x5.reg_eout_li) / len(k9_g1x5.svc_eout_li) )
    k12_g1x5 = SVC_REGULAR(total_trials, 12, 1.5 * np.ones(12))
    que[15] = ("svc rbf better than regular rbf percentage (k=12):", 
            sum(k12_g1x5.svc_eout_li < k12_g1x5.reg_eout_li) / len(k12_g1x5.svc_eout_li) )
    k9_better_k12_ein_percentage = sum(k9_g1x5.reg_ein_li < k12_g1x5.reg_ein_li) / len(k9_g1x5.reg_ein_li)
    k9_better_k12_eout_percentage = sum(k9_g1x5.reg_eout_li < k12_g1x5.reg_eout_li) / len(k9_g1x5.reg_eout_li)
    pretty_table = tabulate(
            [[k9_better_k12_ein_percentage, k9_better_k12_eout_percentage]],
            headers=["k=9 ein < k=12 ein percentage", "k=9 eout < k=12 eout percentage"])
    table = [ [ np.mean(error_li) 
        for error_li in [svc_regular.reg_ein_li, svc_regular.reg_eout_li] ]
        for svc_regular in [k9_g1x5, k12_g1x5] ]
    pretty_table2 = tabulate([["k=9"] + table[0], ["k=12"] + table[1]],
            headers=["", "in sample error", "out of sampler error"])
    que[16] = ("regular rbf changing k",
            "\n" + str(pretty_table) \
            + "\n" + str(pretty_table2))
    k9_g2 = SVC_REGULAR(total_trials, 12, 2 * np.ones(12))
    g1x5_better_g2_ein_percentage = sum(k9_g1x5.reg_ein_li < k9_g2.reg_ein_li) / len(k9_g1x5.reg_ein_li)
    g1x5_better_g2_eout_percentage = sum(k9_g1x5.reg_eout_li < k9_g2.reg_eout_li) / len(k9_g1x5.reg_eout_li)
    pretty_table = tabulate(
            [[g1x5_better_g2_ein_percentage, g1x5_better_g2_eout_percentage]], headers=["g=1.5 ein < g=2 ein percentage", "g=1.5 eout < g=2 eout percentage"])
    table = [ [ np.mean(error_li) 
        for error_li in [svc_regular.reg_ein_li, svc_regular.reg_eout_li] ]
        for svc_regular in [k9_g1x5, k9_g2] ]
    pretty_table2 = tabulate([["g=1.5"] + table[0], ["g=2"] + table[1]],
            headers=["", "in sample error", "out of sampler error"])
    que[17] = ("regular rbf changing gammas", 
            "\n" + str(pretty_table) \
            + "\n" + str(pretty_table2))
    zero_ein = k9_g1x5.reg_ein_li < 1 / (10 * total_trials )
    que[18] = ("regular rbf (k=9, gamma=1.5) zero in sample error percentage : ", 
            sum(zero_ein) / len(zero_ein))
    return que

ans = {
        1 : 'e', # The dimensionality of the Z space should be 65
        2 : 'exd', 
        3 : 'axd',
        4 : 'd',
        5 : 'a',
        6 : 'b',
        7 : 'axd',
        8 : 'b',
        9 : 'e', #all the statements are wrong although d comes close 
        10 : 'a',
        11 : 'c',
        12 : 'c',
        13 : 'a',
        14 : 'e',
        15 : 'd',
        16 : 'd',
        17 : 'dxc',
        18 : 'a',
        19 : 'b',
        20 : 'dxc',
        }

if __name__ == "__main__":
    main()
