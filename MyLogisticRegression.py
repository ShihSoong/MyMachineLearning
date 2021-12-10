import numpy as np
import math
import matplotlib.pyplot as plt

import MyMachineLearningLib as ml

'''
    This python file is going to build a Logistic Regression
    model, which is trying to fit the vector theta where

    theta = min_theta J(theta)

    J(theta) =  -(1/m)*[sigma^m_(i=1) y^(i)*log(sigmoid(theta,x^(i)))
                + (1-y^(i))log(1-sigmoid(theta,x^(i)))]
    
'''


class LogisticRegressionHelper:

    origin_X_train = np.array([])
    origin_X_test = np.array([])

    theta = np.array([[]], dtype='float')
    X_train = np.array([[]], dtype='float')
    y_train = np.array([[]], dtype='int')
    X_test = np.array([[]], dtype='float')
    y_test = np.array([[]], dtype='int')

    J_list = np.array([])
    grad_list = np.array([])

    is_feature_scaling = 1
    fearture_scaling_mean_vec = np.array([])
    fearture_scaling_range_vec = np.array([])


    def __init__(self, _theta=[]) -> None:
        
        if len(_theta):
            self.theta = np.array(_theta).reshape([len(_theta), 1])
        else:
            self.theta = np.array([])

        self.ResetDataSet()
    

    def ResetDataSet(self):
        
        origin_X_train = np.array([])
        origin_X_test = np.array([])
        X_train = np.array([[]])
        y_train = np.array([[]])
        X_test = np.array([[]])
        y_test = np.array([[]])

        is_feature_scaling = 1
        fearture_scaling_mean_vec = np.array([])
        fearture_scaling_range_vec = np.array([])


    def SetTrainDataSet(self, _X_train, _y_train, _is_feature_scaling = 1):

        X = np.array(_X_train)
        y = np.array(np.around(_y_train), dtype='int')
        if np.max(y) > 1 or np.min(y) < 0:
            print('ERROR! element of y over 1 or under 0!')

        if X.size == 0:
            return

        if X.ndim != 2:
            print('Set training data set failed. Because the dimension of _X_train is wrong!')
            return
        
        m, n = X.shape

        if m != y.size:
            print('Set training data set failed. Because the row number of _X_train is not equal to _y_train!')
            return

        if n != len(self.theta):
            print('The row number of _X_train is different to the length of theta, so theta has been reset.')
            self.theta = np.zeros([n, 1])
        
        self.origin_X_train = X
        self.y_train = y.reshape([m, 1])

        if self.X_test.size > 0 and self.origin_X_test.shape[1] != self.origin_X_train.shape[1]:
            self.origin_X_test = np.array([])
            self.X_test = np.array([])
            print('The struct of training set has been changed, so the test set is reseted.')

        self.X_train = np.array(self.origin_X_train)
        self.is_feature_scaling = _is_feature_scaling

        if self.is_feature_scaling:
            self.fearture_scaling_range_vec, self.fearture_scaling_mean_vec = ml.GetFeatureScalingParamter(self.origin_X_train)
            self.X_train = ml.FeatureScalingByParamter(self.origin_X_train, self.fearture_scaling_range_vec, self.fearture_scaling_mean_vec)
        
        if self.X_test.size > 0:
            self.SetTestDataSet(self.origin_X_test, self.y_test)
        
        self.X_train = np.c_[np.ones([self.X_train.shape[0],1]),self.X_train]
                

    def SetTestDataSet(self, _X_test, _y_test):

        X = np.array(_X_test)
        y = np.array(np.around(_y_test), dtype='int')

        if np.max(y) > 1 or np.min(y) < 0:
            print('ERROR! element of y over 1 or under 0!')

        if X.size == 0:
            return

        if X.ndim != 2:
            print('Set training data set failed. Because the dimension of _X_test is wrong!')
            return
        
        m, n = X.shape

        if m != y.size:
            print('Set test data set failed. Because the row number of _X_test is not equal to _y_test!')
            return
        if n != len(self.theta):
            print('Set test data set failed. Because the column number of _X_test is not equal to _X_train!')
            return

        
        self.origin_X_test = X
        self.X_test = np.array(X)
        self.y_test = y.reshape([m, 1])

        if self.is_feature_scaling:
            self.X_test = ml.FeatureScalingByParamter(self.origin_X_test, self.fearture_scaling_range_vec, self.fearture_scaling_mean_vec)

        self.X_test = np.c_[np.ones([self.X_test.shape[0],1]), self.X_test]
    

    def SetTheta(self, _theta):
        
        theta_temp = np.array(_theta).reshape(len(_theta), 1)

        if theta_temp.size != self.X_train.shape[1]:
            print('The size of _theta is different to the column number of X_train, so the training and test set is reseted.')
            self.ResetDataSet()
        
        self.theta = theta_temp
    

    def SetDataSet(self, _X, _y, test_rate, _is_feature_scaling = 1):

        X = np.array(_X)
        y = np.array(_y)

        X_train_temp, y_train_temp, X_test_temp, y_test_temp = ml.SplitDataSet(X, y, test_rate)
        self.SetTrainDataSet(X_train_temp, y_train_temp, _is_feature_scaling)
        self.SetTestDataSet(X_test_temp, y_test_temp)


    def SetDataSetAll(self, _X_y, test_rate, _is_feature_scaling = 1):

        X_y = np.array(_X_y)
        self.SetDataSet(X_y[:,:-1], X_y[:,-1:], test_rate, _is_feature_scaling)
    

    def CostFunc(self, theta, X, y):
        '''
        This function is trying to compute the J(theta) and the gradient.

        X include m examples and every example have n features

        theta should be a column vector that have n rows.
        X should be a matrix that have m rows and n columns, every row is an example 
        and every column is the value of feature.
        y should be a column vector which have m rows.
        '''

        theta = np.array(theta)
        X = np.array(X)
        y = np.array(y)

        m, n = X.shape
        
        #J = 1/m*(-y'*log(sigmoid(X*theta)) - (1-y)'*(log(1-sigmoid(X*theta))));
        J = (1.0 / m) * (   -np.matmul(y.T,ml.MatrixLog(ml.Sigmoid(np.matmul(X,theta))))
                            -np.matmul((1-y).T,ml.MatrixLog(1-ml.Sigmoid(np.matmul(X,theta))))   )
        
        #grad = 1/m * X'*(sigmoid(X*theta) - y);
        grad = (1.0/m) * (np.matmul(X.T,ml.Sigmoid(np.matmul(X,theta)) - y))

        return (J, grad)


    def GradientDescent(self, X, y, _theta, alpha = 1, ebsilon = 0.001):

        J_list = np.array([])
        grad_list = np.array([])
        theta = np.array(_theta)

        J_old, grad = self.CostFunc(theta, X, y)
        J_list = np.append(J_list, J_old)
        grad_list = np.append(grad_list, grad)
        theta = theta - alpha * grad

        J, grad = self.CostFunc(theta, X, y)
        J_list = np.append(J_list, J)
        grad_list = np.append(grad_list, grad)
        theta = theta - alpha * grad

        while(abs(J - J_old) >= ebsilon):
            J_old = J
            J, grad = self.CostFunc(theta, X, y)
            J_list = np.append(J_list, J)
            grad_list = np.append(grad_list, grad)
            theta = theta - alpha * grad
        
        return theta, J_list, grad_list


    def Training(self, alpha = 1, ebsilon = 0.001):

        if self.X_train.size == 0 or self.y_train.size == 0:
            print('ERROR! The traing set is broken!')
            return
        
        if self.X_train.shape[0] != self.y_train.shape[0]:
            print('ERROR! The row number of _X_train is not equal to _y_train!')
            return
        
        if self.X_train.shape[1] != self.theta.size:
            print('WARNING! Theta is not well shaped, so it has been reset.')
            self.theta = np.zeros([self.X_train.shape[1], 1])
        
        self.theta, self.J_list, self.grad_list = self.GradientDescent(
            self.X_train,
            self.y_train,
            self.theta,
            alpha,
            ebsilon
            )

    
    def Predict(self, _X):

        X = np.array(_X)
        if X.size == 0:
            return np.array([])

        if self.is_feature_scaling:
            X = ml.FeatureScalingByParamter(X, self.fearture_scaling_range_vec, self.fearture_scaling_mean_vec)

        X = np.c_[np.ones([X.shape[0],1]),X]
        y = np.array(np.matmul(X, self.theta) > 0, dtype='int')

        return y


    def GetAccuracy(self):

        if self.X_test.size == 0:
            print('ERROR! The test set is empty!')
            return 0
        
        y_pre = np.array(np.matmul(self.X_test, self.theta) > 0, dtype='int')
        y_res = (y_pre == np.array(self.y_test))

        return len(np.where(y_res == True)[0]) / self.y_test.size

    
    def GetFPR(self):

        if self.X_test.size == 0:
            print('ERROR! The test set is empty!')
            return 0

        y_pre = np.array(np.matmul(self.X_test, self.theta) > 0, dtype='int')
        y_ans = np.array(self.y_test, dtype='int')

        TP = len(np.where(y_pre + y_ans == 2)[0])
        TN = len(np.where(y_pre + y_ans == 0)[0])
        FN = len(np.where(y_ans - y_pre == 1)[0])
        FP = len(np.where(y_pre - y_ans == 1)[0])

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = (2 * P * R) / (P + R)

        return (F1, P ,R)
