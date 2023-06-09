"""
Using only numpy and train_test_split from the sklearn library

+ Load data from the dataXX.txt  Suggest and split this data to training data and test data,
2. Propose a linear parametric Model 1. Identify model parameters using the least squares method for training data,
3. Verify the quality of Model 1,
4. Suggest a more complex parametric Model 2. Identify model parameters using the least squares method for training data,
5. Verify the quality of Model 2
6. Compare both models.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def least_square(Xi, yi, n, param=0):
    sum_xi_yi = sum(np.multiply(Xi, yi))
    sum_xi2 = sum(np.power(Xi, 2))
    if param == 0:
        w0 = sum_xi_yi / sum_xi2
        #return w0 * Xi
        return w0
    else:
        w0_nominator = sum(yi) * sum_xi2 - sum(Xi) * sum_xi_yi
        denominator = n * sum_xi2 - sum(Xi) ** 2
        w0 = w0_nominator / denominator
        w1_nominator = n * sum_xi_yi - sum(Xi) * sum(yi)
        w1 = w1_nominator / denominator
        #return w1 * Xi + w0
        weights = [w1, w0]
        return weights


def rmse(y_hat, yi):
    return (sum(np.power(np.subtract(y_hat, yi), 2)) / len(y_hat)) ** 0.5


def non_linear(Xi, yi):
    first = np.linalg.inv(np.array([
        [np.sum(np.power(Xi, 4)), np.sum(np.power(Xi, 3)), np.sum(np.power(Xi, 2))],
        [np.sum(np.power(Xi, 3)), np.sum(np.power(Xi, 2)), np.sum(Xi)],
        [np.sum(np.power(Xi, 2)), np.sum(Xi), len(Xi)]
    ]))
    second = np.array([np.sum(np.multiply(np.power(Xi, 2), yi)), np.sum(np.multiply(Xi, yi)), np.sum(yi)],
                      dtype=np.float64)
    ws = np.matmul(first, second)
    return ws


for i in range(1, 5):
    a = np.loadtxt(f'D:\\polina\\pjatk\\wma\\WMA-pro\\dane\\dane{i}.txt')

    X = a[:, [0]]
    y = a[:, [1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=3)

    weight = least_square(X_train, y_train, len(X_train))
    weights = least_square(X_train, y_train, len(X_train), 1)

    y_1 = weight * X_train
    y_2 = weights[0] * X_train + weights[1]

    plt.plot(X_train, y_train, 'ro')
    plt.plot(X_train, y_1)
    plt.plot(X_train, y_2)
    plt.legend(["data", "model 1", "model 2"])
    plt.show()

    w = non_linear(X_train, y_train)
    y_3 = w[0] * np.power(X_train, 2) + w[1] * X_train + w[2]

    plt.plot(X_train, y_train, 'ro')
    plt.scatter(X_train, y_3)
    plt.legend(["data", "predictions"])
    plt.show()

    y_1_test = weight * X_test
    y_2_test = weights[0] * X_test + weights[1]
    y_3_test = w[0] * np.power(X_test, 2) + w[1] * X_test + w[2]

    print(f'-------------DATA{i}-----------------\n'
          f'rmse model 1: {rmse(y_1, y_train)}, test {rmse(y_1_test, y_test)}\n'
          f'rmse model 2: {rmse(y_2, y_train)}, test {rmse(y_2_test, y_test)}\n'
          f'rmse non linear: {rmse(y_3, y_train)}, test {rmse(y_3_test, y_test)})\n'
          f'-------------------------------------\n\n')
