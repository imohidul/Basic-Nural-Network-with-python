import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


df = pd.read_csv("Data_Classification_7.csv")
data = df.values
m, n = data.shape
X = data[:, 0: n-1]
Y = data[:, n-1]
Y = Y.reshape(-1, 1)
alpha = 0.001
X = np.delete(X, 8, 1)
X = np.delete(X, 11, 1)
X = np.delete(X, 11, 1)
X = np.delete(X, 16, 1)
X = np.delete(X, 16, 1)
row, column = X.shape

#concatenate one vector
ones = np.ones(shape=(row, 1))
X = np.concatenate((ones, X), axis=1)

for i in range(1, X.shape[1]):
        X[:, i] = (X[:, i] - np.amin(X[:, i]))/(np.amax(X[:, i]) - np.amin(X[:, i]))

neurons = int(input("Enter the number neuron in hidden layer : "))

W_h = np.zeros((neurons, column+1))
W_o = np.array([np.random.rand(neurons)])


for i in range(0, neurons):
    W_h[i, :] = np.array([np.random.rand(column+1)])

W_h = W_h.T
W_o = W_o.T


X = np.matrix(X)
W_h = np.matrix(W_h)
Y = np.matrix(Y)

max_iter = int(input("Number of iteration :"))
iteration = 0

while iteration < max_iter:
    Z_h = X * W_h
    A_h = np.tanh(Z_h)
    bias = np.array([np.random.rand(1)])

    Z_o = A_h * W_o + bias
    y = 1 / (1 + np.exp(-1 * Z_o))

    for i in range(row):
        if y[i] < .5:
            np.floor(y[i])
        else:
            np.ceil(y[i])

    E = y - Y

    T_A_h = A_h.T
    dbias = np.zeros((1, 1))
    dW_o = np.zeros((neurons, 1))
    for i in range(row):
        dW_o += T_A_h[:, i] * E[i]
        dbias += E[i]
    W_o -= alpha * dW_o / row
    bias -= alpha * dbias / row

    dW_h = np.zeros((column + 1, neurons))

    for i in range(row):
        dW_h += ((np.multiply(W_o * E[i], (1 - np.square(A_h[i, :])).T)) * X[i, :]).T

    W_h -= alpha * dW_h / row

    iteration += 1




