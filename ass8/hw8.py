import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir_path = os.path.dirname(file_dir)
sys.path.insert(0, tools_dir_path)

from tools import *

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np 


def polykernel():
    # polysvm = SVM(kernel='poly', degree
    pass
    
ans = {
        1 : 'a',
        2 : 'e', 
        3 : 'a',
        4 : 'a',
        5 : '',
        6 : '',
        7 : '',
        8 : '',
        9 : '',
        10 : '',
        }
    

def main():
    output(simulations)

def simulations():
    que = {}
    training_data = np.genfromtxt(os.path.join(file_dir, "features.train"))
    testing_data = np.genfromtxt(os.path.join(file_dir, "features.test"))
    training_set = DataML((training_data[:,1:], 
        np.array(training_data[:,0], dtype="int")))
    # print(training_set)
    polysvm = SVC(kernel='poly', degree=2, C=0.1)
    polysvm.fit(training_set.z, training_set.y)
    print(polysvm.n_support_)
    # print(len(training_set.z))
    training_predicted = polysvm.predict(training_set.z)
    # print(sum(training_predicted == training_set.y))
    confusion = confusion_matrix(training_set.y, training_predicted)
    # print(sum(training_set.y == 8))
    # print(confusion)
    print(classification_report(training_set.y, training_predicted))
    return que

if __name__ == "__main__":
    main()
