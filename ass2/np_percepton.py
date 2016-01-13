import numpy as np

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

def check_classification(data, lin_target_function):
    import matplotlib.pyplot as plt
    plt.plot([-1, 1], [lin_target_function(-1), lin_target_function(1)], '-')
    plt.show()


def main():
    raw_data = n_random_datapoint(100)
    m, c = rand_line()
    print(m ,c)
    target_function = create_lin_target_function(m, c)
    check_classification(0,target_function)

if __name__ == "__main__":
    main()
