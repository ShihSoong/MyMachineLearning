from ctypes import POINTER
from matplotlib import colors
import numpy as np
import MyBPNeuralNetworks as nn
from scipy.io import loadmat
import matplotlib.pyplot as plt

def test_and():
    N = nn.BPNeuralNetwoksHelper([2,1])
    X = np.array([[0,0],[0,1],[1,0],[1,1]],dtype='float')
    # N.Thetas.append(np.array([[-30],[20],[20]]))
    N.Thetas[0] = np.array([[-30],[20],[20]])
    # print(N.ForwardPropagation(X))
    printNN(N)
    printAZ(N.ForwardPropagation(X))

def test_xor():
    N = nn.BPNeuralNetwoksHelper([2,2,1])
    X = np.array([[0,0],[0,1],[1,0],[1,1]],dtype='float')
    # N.Thetas.append(np.array([[-30,10],[20,-20],[20,-20]]))
    # N.Thetas.append(np.array([[-10],[20],[20]]))
    N.Thetas[0] = np.array([[-30,10],[20,-20],[20,-20]])
    N.Thetas[1] = np.array([[-10],[20],[20]])
    # print(N.ForwardPropagation(X))
    printNN(N)
    printAZ(N.ForwardPropagation(X))

def test_BP():
    N = nn.BPNeuralNetwoksHelper([2,2,1])
    X = np.array([[0,0],[0,1],[1,0],[1,1]],dtype='float')
    Y = np.array([[1],[0],[0],[1]], dtype = 'float')
    # N.Thetas.append(np.array([[-30,10],[20,-20],[20,-20]]))
    # N.Thetas.append(np.array([[-10],[20],[20]]))
    N.Thetas[0] = np.array([[-30,10],[20,-20],[20,-20]])
    N.Thetas[1] = np.array([[-10],[20],[20]])
    J, grad = N.Backpropagation(X, Y, 0)
    print('j = {}'.format(J))
    for i in range(len(grad)):
        print('the {0} th grad is \n{1}'.format(i, grad[i]))

def printNN(N):
    print('info of NN:')
    print('the shape is:')
    print(N.shape)
    print('the thetas:')
    for i in range(len(N.Thetas)):
        print('the {0} th of Theta matrix is \n {1}'.format(i, N.Thetas[i]))
    
def printAZ(arg):
    a = arg[0]
    a_list = arg[1]
    z_list = arg[2]
    print('the res is: {}'.format(a))
    print('the a_list is:')
    for i in range(len(a_list)):
        print('the {0} th a is \n{1}'.format(i, a_list[i]))
    
    for i in range(len(z_list)):
        print('the {0} th z is \n{1}'.format(i, z_list[i]))


# test_and()
# test_xor()

# test_BP()

# m = loadmat('ex4data1.mat')
# print(m['y'])
# print(type(m['y']))

def handle_data():
    data = loadmat('ex4data1.mat')
    X = data['X']
    y = data['y']
    Y = np.zeros([y.size, 10], dtype = 'int')
    for i in range(y.size):
        Y[i, (y[i] + 9) % 10] = 1
    # print(Y)
    np.savez('data_for_nn', X = X, Y = Y)

def test1():
    data = np.load('data_for_NN.npz')
    thetas = np.load('Thetas.npz')
    X = np.array(data['X'])
    Y = np.array(data['Y'])
    m, n = X.shape
    NN = nn.BPNeuralNetwoksHelper([n, 25, 10])
    NN.Thetas = [thetas['Theta1'], thetas['Theta2']]
    J, grad = NN.Backpropagation(X, Y, 0)
    print('j = {}'.format(J))
    for i in range(len(grad)):
        print('the {0} th grad is \n{1}'.format(i, grad[i]))

def test2():
    data = np.load('data_for_NN.npz')
    _Thetas = np.load('Thetas.npz')
    Thetas = [_Thetas['Theta1'], _Thetas['Theta2']]
    X = np.array(data['X'])
    Y = np.array(data['Y'])
    J, grad = nn.Backpropagation(X, Y, Thetas, 3)
    print('J = {}'.format(J))
    print('grad = ')
    print(grad)
    print('the nungrad :')
    print(nn.NumGradOnJ(X, Y, Thetas, 0))

# test2()

# handle_data()

def test3():
    data = loadmat('debug_nn_data')
    X = data['X']
    y = data['y']
    Thetas = [np.array(data['Theta1'].T, dtype='float'), np.array(data['Theta2'].T, dtype='float')]
    Y = np.zeros([y.size, 3], dtype = 'int')
    for i in range(y.size):
        Y[i, y[i]-1] = 1
    J, grad = nn.Backpropagation(X, Y, Thetas, 3)
    print('J = {}'.format(J))
    print('grad = ')
    print(grad)
    print('the numgrad :')
    print(nn.NumGradOnJ(X, Y, Thetas, 3))

# test3()

def handle_tehtas():
    Thetas = loadmat('ex4weights.mat')
    Theta1 = Thetas['Theta1'].T
    Theta2 = Thetas['Theta2'].T
    print(Theta1.shape)
    print(Theta1)
    print(Theta2.shape)
    print(Theta2)
    np.savez('Thetas.npz', Theta1 = Theta1, Theta2 = Theta2)

def test4():
    data = np.load('data_for_NN.npz')
    _Thetas = np.load('Thetas.npz')
    Thetas = [_Thetas['Theta1'], _Thetas['Theta2']]
    X = np.array(data['X'])
    Y = np.array(data['Y'])
    N = nn.BPNeuralNetwoksHelper()
    N.SetShape([400,25,10])
    N.SetData(X, Y, 0.3)
    N.RandomInitThetas(-2, 2)
    J_list = N.Train(alpha=10,ebsilon=0.000001,iter_limit=200, lam=1)
    print(J_list)
    print(N.Thetas)
    print('the accuracy = {}'.format(N.GetAccuracy()))
    N.Save('MyNN2')

def test5():
    N = nn.BPNeuralNetwoksHelper()
    N.Load('MyNN.npz')
    # N.SetX_test(N.X_train)
    # N.SetY_test(N.Y_train)
    print('the accuracy = {}'.format(N.GetAccuracy(N.X_test, N.Y_test)))
    # N.Save('MyNN')

def test6():
    data = np.load('data_for_NN.npz')
    _Thetas = np.load('Thetas.npz')
    Thetas = [_Thetas['Theta1'], _Thetas['Theta2']]
    X = np.array(data['X'])
    Y = np.array(data['Y'])
    N = nn.BPNeuralNetwoksHelper()
    N.SetShape([400,25,10])
    N.SetData(X, Y, 0.3)
    N.RandomInitThetas(-2, 2)
    J_list, train_acc, test_acc = N.Train(alpha=10, ebsilon=0.000001, iter_limit=500, lam=1, get_accuracy=True)
    print(J_list)
    print(N.Thetas)
    print('the accuracy = {}'.format(N.GetAccuracyOnTest()))
    # plt.plot(train_acc, color = 'y')
    # plt.plot(test_acc, color = 'b')
    plt.plot(J_list)
    plt.show()

test6()