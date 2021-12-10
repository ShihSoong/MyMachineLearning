import MyLogisticRegression as lr
import numpy as np
import matplotlib.pyplot as plt
import MyMachineLearningLib as ml

def LoadData():
    data = np.loadtxt('data1.txt', delimiter = ',')
    X = data[...,:-1]
    y = data[...,-1:]

    return (X, y)

def PlotSigmoid(x_range, theta_val):

    y = ml.Sigmoid(theta_val * x_range)
    plt.plot(x_range,y)


def TestSigmoid():
    '''
    this function is going to test sigmoid function
    '''
    theta = np.array([  [-3],
                        [2]])
    x = np.array([  [1,2],
                    [2,3],
                    [4,5]])

    print('theta = {0}'.format(theta))
    print('x = {0}'.format(x))
    print('result = {0}'.format(ml.Sigmoid(np.matmul(x,theta))))
    print('result = {0}'.format(ml.Sigmoid(np.matmul(x,theta).T)))

    PlotSigmoid(np.linspace(-10,10,100),1)
    print('the shape of the plot should be like letter "Z"')

def test():
    l = lr.LogisticRegressionHelper()
    data = np.loadtxt('data1.txt', delimiter = ',')
    
    l.SetDataSetAll(data, 0.1, 1)
    l.Training(ebsilon=0.000001)
    print(l.theta, l.J_list)

def PlotData(X, y):

    pos_index = np.nonzero(y)[0]
    neg_index = np.where(y==0)[0]

    X_pos = X[pos_index]
    X_neg = X[neg_index]
    
    plt.scatter(X_pos[...,1],X_pos[...,2],c='b')
    plt.scatter(X_neg[...,1],X_neg[...,2],c='r')

def PlotData2(X, y):

    pos_index = np.nonzero(y)[0]
    neg_index = np.where(y==0)[0]

    X_pos = X[pos_index]
    X_neg = X[neg_index]
    
    plt.scatter(X_pos[...,0],X_pos[...,1],c='b')
    plt.scatter(X_neg[...,0],X_neg[...,1],c='r')

def PlotThetaLine(theta, start, end):
    x_range = np.linspace(start, end, 30)
    y_range = (-theta[1] * x_range - theta[0]) / theta[2]

    plt.plot(x_range,y_range)

def test2():
    #TestSigmoid()
    X, y = LoadData()
    X = np.c_[np.ones([X.shape[0],1]),X]
    # PlotData(X, y)
    X[:,1:] = ml.FeatureScaling(X[:,1:])[0]
    PlotData(X, y)
    print('X = ')
    print(X)

    theta = np.array([[0],[0],[0]])
    l = lr.LogisticRegressionHelper()
    theta, J_list, grad_list = l.GradientDescent(X, y, theta, 1, ebsilon=0.000001)
    print('theta = ')
    print(theta)
    print('J = {}'.format(J_list))
    print('grad = {}'.format(grad_list))
    PlotThetaLine(theta,0,1)
    # plt.plot(J_list)
    plt.show()

def test3():

    l = lr.LogisticRegressionHelper()
    data = np.loadtxt('data1.txt', delimiter = ',')
    
    l.SetDataSetAll(data, 0.3)

    PlotData(l.X_train, l.y_train)

    l.Training(alpha = 10, ebsilon=0.000001)

    print(l.J_list)
    PlotThetaLine(l.theta, 0, 1)

    print('accuracy : {}'.format(l.GetAccuracy()))
    print('F1 P R : {}'.format(l.GetFPR()))

    plt.show()

test3()
