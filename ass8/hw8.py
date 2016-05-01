import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir_path = os.path.dirname(file_dir)
sys.path.insert(0, tools_dir_path)

from tools import *

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import cross_val_score, KFold

import numpy as np 


def allexcept(digit, *col_data_sets):
    copies_of_data_sets = [ DataML((data_set.z, data_set.y)) for data_set in col_data_sets ]
    for data_set in copies_of_data_sets:
        truth_table = data_set.y != digit 
        data_set.y[truth_table] = -1
    return copies_of_data_sets

def trial_all_except(training_set, testing_set, digit, kernel, c, degree=None):
    training_set, testing_set = allexcept(digit, training_set, testing_set)
    if degree == None:
        svm = SVC(kernel=kernel, C=c)
    else:
        svm = SVC(kernel=kernel, C=c, degree=degree)
    svm.fit(training_set.z, training_set.y)
    training_predicted = svm.predict(training_set.z)
    in_sample_error = classified_error(training_predicted, training_set.y)
    testing_predicted = svm.predict(testing_set.z)
    out_of_sample_error = classified_error(testing_predicted, testing_set.y)
    confusion = confusion_matrix(training_set.y, training_predicted)
    report = classification_report(training_set.y, training_predicted)
    return svm.n_support_, (in_sample_error, out_of_sample_error), report, confusion

def a_vs_b(a, b, *col_data_sets):
    copies_of_data_sets = [ DataML((data_set.z, data_set.y)) for data_set in col_data_sets ]
    for i in range(len(copies_of_data_sets)):
        data_set = copies_of_data_sets[i]
        truth_a = data_set.y == a
        truth_b = data_set.y == b
        z = np.vstack([data_set.z[truth_a], data_set.z[truth_b]])
        y = np.concatenate([data_set.y[truth_a], data_set.y[truth_b]])
        copies_of_data_sets[i] = DataML((z,y))
    return copies_of_data_sets


def trial_a_vs_b(training_set, testing_set, a, b, kernel, c, degree=None):
    training_set, testing_set = a_vs_b(a, b, training_set, testing_set)
    if degree == None:
        svm = SVC(kernel=kernel, C=c)
    else:
        svm = SVC(kernel=kernel, C=c, degree=degree)
    svm.fit(training_set.z, training_set.y)
    training_predicted = svm.predict(training_set.z)
    in_sample_error = classified_error(training_predicted, training_set.y)
    testing_predicted = svm.predict(testing_set.z)
    out_of_sample_error = classified_error(testing_predicted, testing_set.y)
    confusion = confusion_matrix(training_set.y, training_predicted)
    report = classification_report(training_set.y, training_predicted)
    return svm.n_support_, (in_sample_error, out_of_sample_error), report, confusion

def best_c(training_set):
    training_set = a_vs_b(1, 5, training_set)[0]
    svcs = [ SVC(kernel='poly', C=c, degree=2)
            for c in [0.0001, 0.001, 0.01, 0.1, 1] ]
    cv = KFold(n=len(training_set.y), n_folds=10, shuffle=True)
    score_c = [ np.mean(
        cross_val_score(polysvm, training_set.z, training_set.y, cv=cv))
        for polysvm in svcs ]
    return np.argmax(score_c), score_c

ans = {
        1 : 'axd',
        2 : 'a', 
        3 : 'a',
        4 : 'c',
        5 : 'a&d', # both true
        6 : 'b',
        7 : 'c', # disagrees with given answer. Perhaps due to difference in svm libraries used
        8 : 'c',
        9 : 'e', 
        10 : 'c&d', # tied for lowest eout
        }
    

def main():
    output(simulations)

def simulations():
    que = {}
    training_data = np.genfromtxt(os.path.join(file_dir, "features.train"))
    testing_data = np.genfromtxt(os.path.join(file_dir, "features.test"))
    def convert_raw(t_data):
        return DataML((t_data[:,1:], np.array(t_data[:,0], dtype="int")))
    training_set = convert_raw(training_data)    # print(training_set)
    testing_set = convert_raw(testing_data)

    results_even = [ trial_all_except(training_set, testing_set, digit, 'poly', 0.1, 2)
            for digit in range(0,9,2) ]
    in_sample_error_list_even = [ result[1][0] for result in results_even ]
    que[2] = ("digit with highest in sample error :", (np.argmax(in_sample_error_list_even) * 2 , np.max(in_sample_error_list_even)) )
    results_odd = [ trial_all_except(training_set, testing_set, digit, 'poly', 0.1, 2)
            for digit in range(1,10,2) ]
    in_sample_error_list_odd = [ result[1][0] for result in results_odd ]
    que[3] = ("digit with lowest in sample error :", (np.argmin(in_sample_error_list_odd) * 2 + 1 , np.min(in_sample_error_list_odd)) )
    support_vector_difference = abs(
            sum(results_even[np.argmax(in_sample_error_list_even)][0])
            - sum(results_odd[np.argmin(in_sample_error_list_odd)][0]))
    que[4] = ("support vector difference :", support_vector_difference)
    results = [ trial_a_vs_b(training_set, testing_set, 1, 5, 'poly', c, 2) 
            for c in [0.001, 0.01, 0.1, 1] ]
    support_vectors = [ sum(result[0]) for result in results ]
    out_of_sample_errors = [ result[1][1] for result in results ]
    in_sample_errors = [ result[1][0] for result in results ]
    que[5] = ("various stats", 
    "\n\tsupport vectors\n\t" + str(support_vectors)
    + "\n\tout of sample errors\n\t" + str(out_of_sample_errors)
    + "\n\tin sample errors\n\t" + str(in_sample_errors)
    )
    results = [ [trial_a_vs_b(training_set, testing_set, 1, 5, 'poly', c, degree) 
            for c in [0.0001, 0.001 ,0.01, 0.1, 1]] for degree in range(2,6) ]
    results_transpose = [ [results[i][j] for i in range(len(results)) ] 
            for j in range(len(results[0])) ]
    c_lowest_ein =  [ result[1][0] for result in results_transpose[0] ]
    support_vectors = [ sum(result[0]) for result in results_transpose[1] ]
    c_third_lowest_ein =  [ result[1][0] for result in results_transpose[0] ]
    c_highest_eou = [ result[1][1] for result in results_transpose[-1] ]
    que[6] = ("various stats", 
    "\n\tin sample errors when c = 0.0001\n\t" + str(c_lowest_ein)
    + "\n\tsupport vectors when c = 0.001\n\t" + str(support_vectors)
    + "\n\tin sample errors when c = 0.01\n\t" + str(c_third_lowest_ein)
    + "\n\tout of sample errors when c = 1\n\t" + str(c_highest_eou)
    )
    results = [ best_c(training_set) for _ in range(50) ]
    frequency = np.bincount([ result[0] for result in results ])
    que[7] = ("frequency :", frequency)
    best = np.argmax(frequency)
    average_score_of_best = np.mean([ result[1][best] for result in results ])
    que[8] = ("average_score_of_best :", average_score_of_best)
    results = [ trial_a_vs_b(training_set, testing_set, 1, 5, 'rbf', c)
            for c in [0.01, 1, 100, 10**4, 10**6 ] ]
    in_sample_errors = [ result[1][0] for result in results ] 
    que[9] = ("in sample errors :", in_sample_errors)
    out_of_sample_errors = [ result[1][1] for result in results ] 
    que[10] = ("out of sample errors :", out_of_sample_errors)
    return que

if __name__ == "__main__":
    main()
