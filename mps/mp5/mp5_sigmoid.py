"""
Taking the data from linear regression task - data is ordered according list.

1. Translate matlab code (or octave, scilab) to python code.
2. Use another activation function, eg arctang
"""

import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt('D:\\polina\\pjatk\\wma\\WMA-pro\\dane\\dane12.txt')

P = np.array(a[:, 0])
T = a[:, [1]]

P = np.reshape(P, (51, 1)).T
T = np.reshape(T, (51, 1)).T

# Network Initialization
S1 = 2
W1 = np.random.random((S1, 1)) - 0.5
B1 = np.random.random((S1, 1)) - 0.5
W2 = np.random.random((1, S1)) - 0.5
B2 = np.random.random((1, 1)) - 0.5
lr = 0.001

for epoch in range(1500):
    # Forward propagation
    X = np.dot(W1, P) + np.dot(B1, np.ones(P.shape))
    # A1 = 1 / (1 + np.power(np.e, X))
    A1 = np.arctan(X)
    A2 = np.dot(W2, A1) + B2

    # Backpropagation
    E2 = T - A2
    E1 = np.dot(W2.T, E2)

    dW2 = lr * np.dot(E2, A1.T)
    dB2 = lr * np.dot(E2, np.ones(E2.shape).T)

    derivative = np.divide(1, 1 + np.power(A2, 2))
    dW1 = lr * np.dot(derivative, P.T)
    dB1 = lr * np.dot(derivative, np.ones(P.shape).T)

    W2 += dW2
    B2 += dB2
    W1 += dW1
    B1 += dB1

    if epoch % 100 == 0:
        plt.clf()
        plt.plot(P, T, 'r+')
        plt.plot(P, A2, 'y*')
        plt.pause(0.1)

plt.show()
