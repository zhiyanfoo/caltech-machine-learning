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
    z = [ [z1(x), z2(x)] for x in coordinates ]
    return z

print(svm_que_helper())
