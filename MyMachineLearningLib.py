import numpy as np
import math
import matplotlib.pyplot as plt
import random

from numpy.core.fromnumeric import mean

def Sigmoid(z_):
    '''
    theta should be a real number or a column vector
    x should be a real number or a column vector
    example:
        theta = [   [-3],
                    [2]]
        x =     [   [1],
                    [2]]
        then the z = theta.T * x = 1
        thus, the result should be 0.73529...
    '''

    z = np.array(z_)
    origin_shape = z.shape
    z = z.reshape(z.size)
    #print(z)
    h = np.zeros(z.size)
    
    for i in range(z.size):
        h[i] = 1.0 / (1 + math.exp(-z[i]))
    
    h = h.reshape(origin_shape)

    return h

def GradientSigmoid(_z):
    return Sigmoid(_z) * (1 - Sigmoid(_z))

def MatrixLog(X_):
    
    X = np.array(X_)
    origin_shape = X.shape
    X = X.reshape(X.size)

    Y = np.zeros(X.size)

    for i in range(X.size):
        Y[i] = math.log(X[i])
    
    Y = Y.reshape(origin_shape)

    return Y

def GetFeatureScalingParamter(_X):
    X = np.array(_X)

    m, n = X.shape

    range_vec = X.max(axis = 0) - X.min(axis = 0)
    mean_vec = X.mean(axis = 0)

    return range_vec, mean_vec

def FeatureScalingByParamter(_X, range_vec, mean_vec):
    X = np.array(_X)
    m, n = X.shape

    for i in range(m):
        for j in range(n):
            X[i][j] = (X[i][j] - mean_vec[j]) / range_vec[j]
    
    return X

def FeatureScaling(x):

    X = np.array(x)

    m, n = X.shape

    range_vec = X.max(axis = 0) - X.min(axis = 0)
    mean_vec = X.mean(axis = 0)

    for i in range(n):
        for j in range(m):
            X[j][i] = (X[j][i] - mean_vec[i]) / range_vec[i] + 0.5
    
    return X, mean_vec, range_vec


def SplitDataSet(_X, _y, test_rate):

    X = np.array(_X)
    y = np.array(_y)

    X_y = np.c_[X, y]
    m = X_y.shape[0]

    rand_index = np.array(range(m))

    random.shuffle(rand_index)

    X_y = X_y[rand_index]

    m_test = int(np.ceil(m * test_rate))
    m_train = int(m - m_test)

    X_train = X_y[0:m_train, 0:-1]
    y_train = X_y[0:m_train, -1:]
    X_test = X_y[m_train:, 0:-1]
    y_test = X_y[m_train:, -1:]

    return (X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    data = np.loadtxt('data1.txt', delimiter = ',')
    X = data[...,:-1]
    y = data[...,-1:]

    print('X = \n',X)
    print('y = \n',y)

    X_train, y_train, X_test, y_test = SplitDataSet(X, y, 0.7)

    print('X_train = \n',X_train)
    print('y_train = \n',y_train)
    print('X_test = \n',X_test)
    print('y_test = \n',y_test)