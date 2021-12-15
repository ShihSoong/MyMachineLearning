import numpy as np
import pandas as pd
import MyLogisticRegression as lr
import MyBPNeuralNetworks as nn
import matplotlib.pyplot as plt

data = pd.read_csv('bank.csv', sep = ',')

data = data.drop(data[data.job == 'unknown'].index)
data = data.drop(data[data.education == 'unknown'].index)
data = data.drop(columns=['contact', 'day', 'month'])

data = data.rename(columns={'y':'deposit'})
data.insert(len(data.columns), 'y', data.default)
data = data.drop(columns=['default'])

print(data.shape)
print(data.head())

job_val = {'unemployed':0, 
           'student':1,
           'retired':2,
           'services':3, 
           'housemaid':4, 
           'blue-collar':5,
           'self-employed':6,
           'technician':7,
           'admin.':8, 
           'management':9, 
           'entrepreneur':10
          }

marital_val = {'divorced':-1,
               'single':0,
               'married':1}

boolean_val = {'yes':1,
               'no':0}

education_val = {'primary':1,
                 'secondary':2,
                 'tertiary':3}

poutcome_val = {'unknown':0,
                'failure':-1,
                'other':0,
                'success':1}

data['job'] = np.array([job_val[x] for x in data['job']], dtype = 'int')
data['marital'] = np.array([marital_val[x] for x in data['marital']], dtype = 'int')
data['education'] = np.array([education_val[x] for x in data['education']], dtype = 'int')
data['poutcome'] = np.array([poutcome_val[x] for x in data['poutcome']], dtype = 'int')

for token in ['housing', 'loan', 'deposit', 'y']:
    data[token] = np.array([boolean_val[x] for x in data[token]], dtype = 'int')

print(data.dtypes)
print(data)

true_set = data[data['y'] == 1].values
false_set = data[data['y'] == 0].values

training_set = np.r_[true_set[:50, :], false_set[:50, :]]
test_set = np.r_[true_set[50:70, :], false_set[80:100, :]]

N = nn.BPNeuralNetwoksHelper([10,30,1])

X_train = training_set[:,:-1]
Y_train = training_set[:,-1:]
X_test = test_set[:,:-1]
Y_test = test_set[:,-1:]

# print(X)
# print(Y)

# N.SetData(X, Y, 0.5)
N.SetX_train(X_train)
N.SetX_test(X_test)
N.SetY_train(Y_train)
N.SetY_test(Y_test)
J, AccTrain, AccTest = N.Train(feature_scaling=True, iter_limit=10000, epsilon=0.00001)
print(N.GetAccuracyOnTest())