import MyMachineLearningLib as ml
import numpy as np
import time
import random
import tqdm

def ForwardPropagation(_X_input, _Thetas):
        
        a_list = list()
        z_list = list()

        a = np.array(_X_input, dtype = 'float')
        a = np.c_[np.ones([a.shape[0], 1]), a]

        a_list.append(a)
        z_list.append(np.array([]))

        for i in range(len(_Thetas)-1):
            z = np.matmul(a, _Thetas[i])
            a = ml.Sigmoid(z)
            a = np.c_[np.ones([a.shape[0], 1]), a]
            z_list.append(np.array(z))
            a_list.append(np.array(a))
        
        z = np.matmul(a, _Thetas[-1])
        a = ml.Sigmoid(z)
        z_list.append(np.array(z))
        a_list.append(np.array(a))
        
        return a, a_list, z_list


def NNCostFunction(m,  _Y_pre, _Y, _lam, _Thetas):

    J_pure = (1 / m) * np.sum(np.sum((-_Y * ml.MatrixLog(_Y_pre) - (1-_Y) * ml.MatrixLog(1 - _Y_pre)), axis=0),axis=0)
    # cost function without regularization

    r = 0
    # cost function regularization term
    for theta in _Thetas:
        r = r + np.sum(np.sum(theta[1:,:] * theta[1:,:], axis=0), axis=0)
    r = r * _lam / 2 / m

    return J_pure + r


def NumGradOnJ(_X, _Y, Thetas, lam, e = 0.0001):
    '''
    This function is used to check the Backpropagation.
    This function is very slow, so it can't replace the BP algrithm.
    Note: because of the speed, the nerwork shouldn't be very large.
    '''

    X = np.array(_X, dtype = 'float')
    Y = np.array(_Y, dtype = 'float')

    Thetas_bigger = []
    Thetas_smaller = []
    grad = []
    m = X.shape[0]

    for i in range(len(Thetas)):
        Thetas_bigger.append(np.array(Thetas[i]))
        Thetas_smaller.append(np.array(Thetas[i]))
        grad.append(np.array(Thetas[i]))

    for i in range(len(Thetas)):
        for j in range(Thetas[i].shape[0]):
            for k in range(Thetas[i].shape[1]):
                Thetas_bigger[i][j, k] = Thetas_bigger[i][j, k] + e
                Thetas_smaller[i][j, k] = Thetas_smaller[i][j, k] - e

                Y_pre_bigger = ForwardPropagation(X, Thetas_bigger)[0]
                Y_pre_smaller = ForwardPropagation(X, Thetas_smaller)[0]

                grad[i][j, k] = (NNCostFunction(m, Y_pre_bigger, Y, lam, Thetas_bigger)
                                - NNCostFunction(m, Y_pre_smaller, Y, lam, Thetas_smaller)) / (2 * e)
                
                Thetas_bigger[i][j, k] = Thetas_bigger[i][j, k] - e
                Thetas_smaller[i][j, k] = Thetas_smaller[i][j, k] + e
    return grad


def Backpropagation(_X, _Y, Thetas, lam):

    X = np.array(_X, dtype = 'float')
    Y = np.array(_Y, dtype = 'float')

    'Initialise the gradient matrix of theta.'
    theta_gradient = list()

    for theta in Thetas:
        theta_gradient.append(np.zeros(theta.shape))
    
    pre_res, a_list, z_list = ForwardPropagation(X, Thetas)
    m = X.shape[0]
    J = NNCostFunction(m, pre_res, Y, lam, Thetas)
    
    delta_list = a_list.copy()
    delta_list[len(Thetas)] = a_list[len(Thetas)] - Y
    for i in range(len(Thetas)-1, 0 ,-1):
        delta_list[i] = np.matmul(delta_list[i+1], Thetas[i].T[:,1:]) * ml.GradientSigmoid(z_list[i])
    # compute the delta

    for i in range(m):
        for j in range(len(theta_gradient)):
            theta_gradient[j] = theta_gradient[j] + np.matmul(a_list[j][i:i+1,:].T,delta_list[j+1][i:i+1,:])
    
    for i in range(len(theta_gradient)):
        theta_gradient[i] = theta_gradient[i] / m
        theta_gradient[i][1:,:] = theta_gradient[i][1:,:] + lam * Thetas[i][1:,:] / m
    
    return J, theta_gradient



class BPNeuralNetwoksHelper:

    shape = np.array([], dtype = 'int')
    '''
    shape is a matrix that include the message of layers.
    For example, if shape[i] = p, that means the i th layer has p units
    '''
    
    X_train = np.array([[]], dtype = 'float')
    '''
    X_train is the input matrix to train the network, it has m rows and n columns.
    m is the number of examples. Every row is an example. 
    n is the number of features. Every column is a feature. Also, n is the number of the neros of 
    input layer.
    '''
    Y_train = np.array([[]], dtype = 'int')
    '''
    Y_train is the output matrix to train the network, it has m rows and p columns.
    m is the number of examples, it must equal to the row number of X_train.
    p is the length of exery output.
    It is strongly recommanded that the every row of Y_train is one-hot vector.
    '''

    X_test  = np.array([[]], dtype = 'float')
    Y_test  = np.array([[]], dtype = 'int')

    Thetas = list()
    '''
    Thetas is a list of matrix.
    Assume n is the number of hidden layer, then the size of Theta should be n+1 and the number 
    of all layer should be n+2(include the input layer and the output layer).
    Theta[i] is the paramter of connecting the i layer and the i+1 layer.
    '''

    is_feature_scaling = False
    feature_scaling_mean_vec = np.array([], dtype = 'float')
    feature_scaling_range_vec = np.array([], dtype = 'float')

    def __init__(self, _shape = np.array([], dtype = 'int')) -> None:
        self.SetShape(_shape)


    def SetShape(self, _shape = np.array([], dtype = 'int')):
        _shape = np.array(_shape, dtype = 'int')
        _shape = _shape.reshape(_shape.size)
        self.shape = _shape
        self.RandomInitThetas()
    

    def RandomInitThetas(self, start = -1, end = 1):
        np.random.seed(int(time.time()))
        self.Thetas.clear()
        for i in range(self.shape.size - 1):
            self.Thetas.append(np.random.random([self.shape[i] + 1,self.shape[i+1]]) * (end - start) + start)


    def SetX_train(self, _X_train):
        self.X_train = np.array(_X_train, dtype = 'float')
        if self.shape[0] != self.X_train.shape[1]:
            self.shape[0] = self.X_train.shape[1]
            self.SetShape(self.shape)
    

    def SetY_train(self, _Y_train):
        self.Y_train = np.array(_Y_train, dtype = 'int')
        if self.shape[-1] != self.Y_train.shape[1]:
            self.shape[-1] = self.Y_train.shape[1]
            self.SetShape(self.shape)
    

    def SetX_test(self, _X_test):
        self.X_test = np.array(_X_test, dtype = 'float')
    

    def SetY_test(self, _Y_test):
        self.Y_test = np.array(_Y_test, dtype = 'int')
    
    
    def SetData(self, _X, _Y, test_reta):

        X = np.array(_X)
        Y = np.array(_Y)
        Data = np.c_[X, Y]
        m, n = X.shape
        m_train = m - int(np.around(m * test_reta))

        rand_index = np.array(range(m))
        random.shuffle(rand_index)
        Data = Data[rand_index]

        self.SetX_train(Data[0:m_train, 0:n])
        self.SetY_train(Data[0:m_train, n:])
        self.SetX_test(Data[m_train:, 0:n])
        self.SetY_test(Data[m_train:, n:])


    def Save(self, filename, version = 1):
        '''
        To save the whole net.

        For the version 1.0:
        version,
        shape, Thetas_vector, X_train, Y_train, X_test, Y_test
        is_feature_scaling, fearture_scaling_mean_vec, feature_scaling_range_vec
        '''
        if version == 1:
            Thetas_vector = np.array([], dtype = 'float')
            for Theta in self.Thetas:
                Thetas_vector = np.append(Thetas_vector, Theta)
            np.savez(filename, 
                    version = version,
                    shape = self.shape, 
                    Thetas_vector = Thetas_vector,
                    X_train = self.X_train, 
                    Y_train = self.Y_train, 
                    X_test = self.X_test, 
                    Y_test = self.Y_test,
                    is_feature_scaling = self.is_feature_scaling,
                    feature_scaling_mean_vec = self.feature_scaling_mean_vec,
                    feature_scaling_range_vec = self.feature_scaling_range_vec)
    

    def Load(self, filename):
        data = np.load(filename)
        
        version = int(data['version'])

        if version == 1:
            self.shape = np.array(data['shape'], dtype = 'float')
            self.X_train = np.array(data['X_train'], dtype = 'float')
            self.Y_train = np.array(data['Y_train'], dtype = 'int')
            self.X_test = np.array(data['X_test'], dtype = 'float')
            self.Y_test = np.array(data['Y_test'], dtype = 'int')
            self.is_feature_scaling = bool(data['is_feature_scaling'])
            self.feature_scaling_mean_vec = np.array(data['feature_scaling_mean_vec'], dtype = 'float')
            self.feature_scaling_range_vec = np.array(data['feature_scaling_range_vec'], dtype = 'float')

            pos = 0
            for i in range(self.shape.size - 1):
                theta_shape = (int(self.shape[i]) + 1, int(self.shape[i + 1]))
                theta_size = int(theta_shape[0] * theta_shape[1])
                self.Thetas.append(np.array(data['Thetas_vector'][pos: pos + theta_size], dtype = 'float').reshape(theta_shape))
                pos = pos + theta_size
        
        else:
            print('ERROR! Version is wrong.')


    def Predict(self, _X_input):
        if self.is_feature_scaling:
            X = ml.FeatureScalingByParamter(_X_input, self.feature_scaling_range_vec, self.feature_scaling_mean_vec)
        else:
            X = _X_input
        return np.around(ForwardPropagation(X, self.Thetas)[0])
    

    def GetAccuracy(self, X, Y):

        if X.size == 0 or Y.size == 0:
            print('ERROR! Test set is empty.')
            return

        if self.is_feature_scaling:
            X = ml.FeatureScalingByParamter(X, self.feature_scaling_range_vec, self.feature_scaling_mean_vec)
        
        y_pre = self.Predict(X)
        m = y_pre.shape[0]
        y_res = np.zeros(m)

        np.savetxt('y_pre_test.csv',y_pre)

        for i in range(m):
            y_res[i] = int(not 0 in np.array((y_pre[i, :] == Y[i, :]),dtype='int'))
        
        return np.sum(y_res) / m
    

    def GetAccuracyOnTest(self):
        return self.GetAccuracy(self.X_test, self.Y_test)
    

    def Train(self, alpha = 1.0, epsilon = 0.001, iter_limit = 100, lam = 0, feature_scaling = False, get_accuracy = False):

        if iter_limit < 2:
            # iter_limit = float('inf')
            print('ERROR! the iter limit is too small.')
            return np.array([])

        self.is_feature_scaling = feature_scaling
        
        if self.is_feature_scaling:
            self.feature_scaling_range_vec, self.feature_scaling_mean_vec = ml.GetFeatureScalingParamter(self.X_train)
            X = ml.FeatureScalingByParamter(self.X_train, self.feature_scaling_range_vec, self.feature_scaling_mean_vec)
        else:
            X = np.array(self.X_train)
        
        J_list = []
        train_accuracy_list = []
        test_accuracy_list = []

        J_old, grad_old = Backpropagation(X, self.Y_train, self.Thetas, lam)
        J_list.append(J_old)
        for i in range(len(self.Thetas)):
            self.Thetas[i] = self.Thetas[i] - alpha * grad_old[i]
        if get_accuracy:
            train_accuracy_list.append(self.GetAccuracy(self.X_train, self.Y_train))
            test_accuracy_list.append(self.GetAccuracy(self.X_test, self.Y_test))
        
        J, grad = Backpropagation(X, self.Y_train, self.Thetas, lam)
        J_list.append(J)
        for i in range(len(self.Thetas)):
            self.Thetas[i] = self.Thetas[i] - alpha * grad[i]
        if get_accuracy:
            train_accuracy_list.append(self.GetAccuracy(self.X_train, self.Y_train))
            test_accuracy_list.append(self.GetAccuracy(self.X_test, self.Y_test))

        # iter_times = 2;
        # while abs(J_old - J) > ebsilon and iter_times < iter_limit:
        for i in tqdm.tqdm(range(iter_limit - 2)):
            J_old = J
            J, grad = Backpropagation(X, self.Y_train, self.Thetas, lam)
            J_list.append(J)
            for i in range(len(self.Thetas)):
                self.Thetas[i] = self.Thetas[i] - alpha * grad[i]
            if get_accuracy:
                train_accuracy_list.append(self.GetAccuracy(self.X_train, self.Y_train))
                test_accuracy_list.append(self.GetAccuracy(self.X_test, self.Y_test))
            if J > J_old:
                alpha = alpha / 2
            # the cost function is becoming larger, so the alpha should be smaller.
            if abs(J_old - J) <= epsilon:
                break
        
        return  (np.array(J_list, dtype = 'float'),
                np.array(train_accuracy_list, dtype = 'float'),
                np.array(test_accuracy_list, dtype = 'float'))
        