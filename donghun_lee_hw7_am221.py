
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', palette='Blues')
from cvxpy import *


columns = ['m1', 'm2', 'm3', 'm4', 'y']
features = columns[:-1]

df = pd.read_csv('banknotes.data.csv', header=None, names=columns)
print(df.head())
print(df.describe())

# initialize data
# 1 = forged, 0 = not forged
y = df.loc[:, 'y'].as_matrix()
yhat = 2 * y - 1
X = df.loc[:, features].as_matrix()


## In[24]:
#
#from hw7 import Perceptron
#df = pd.read_csv('banknotes.data', header=None, names=columns)
#not_forged = df.loc[:, 'y'] == 0
#b = -1*np.ones((df.shape[0], 1))
#df.loc[not_forged, features] = df.loc[not_forged, features].apply(lambda x: x * -1)
#X_modified = np.hstack((X, b))
#p = Perceptron(max_iter=10**5)
#W = p.fit(X_modified)
#print(W)

num_lamdas = 14
lamdas = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000,1e5,1e6]

W_list = []
#Xi_list = []
accuracy_list = []
status_list = []
solution_list = []

n = X.shape[0]
d = X.shape[1]
print('n', n, 'd', d)


for ld in lamdas:
    W = Variable(d)
    b = Variable()
    Xi = Variable(n)

    lamda = Parameter()
    lamda.value = ld


    ones = Parameter(n)
    ones.value = np.ones(n)
    
    # objective
    obj = Minimize(0.5 * square(norm(W, 2)) + lamda * sum_entries(Xi))
    # constraints
    constraints = []
    for i in range(n):
        constraints += [
            yhat[i] * (W.T * X[i] + b) + Xi[i] >= 1,
            Xi[i] >= 0
        ]
    #constraints = [yhat.T *(X * W + b) + Xi >= ones,
    #               Xi >= 0]

    prob = Problem(obj, constraints)
    
    prob.solve()
    status_list.append(prob.status)
    solution_list.append(prob.value)
    print(prob.value)
    W_star = W.value
    print('W', W.value)
    print('b', b.value)
    print('')
    W_list.append(W_star)
    y_hat = np.array(np.dot(X, W_star) + b.value > 0)
    y_hat = y_hat.squeeze(axis=1).astype(int)
    accuracy = np.sum(y_hat.astype(int) == y) / n

    accuracy_list.append(accuracy)
    #Xi_list.append(Xi.value)


best_lamda = np.argmax(accuracy_list)
plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.plot(np.log10(lamdas), accuracy_list, marker='o', linestyle='--', c='g')
ax.scatter(np.log10(lamdas[best_lamda]), accuracy_list[best_lamda], marker='x', c='r', s=500.0)
ax.set_xlabel('Lamda')
ax.set_ylabel('Accuracy')
plt.show()
plt.savefig("temp.png")
