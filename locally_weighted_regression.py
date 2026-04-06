# Locally Weighted Regression
# Dataset: x06Simple.csv (columns: x1, x2, y)
# Dependencies: numpy, sklearn (train_test_split only)
#
# Implementation: For each test point, fits a separate weighted linear regression
# where training points closer to the query point are given higher weights.
# Weights are computed using: w = exp(-L1_distance / k^2)
# where k=1 is the bandwidth parameter. Theta is solved in closed form per query point.
#
# Output: RMSE on test set

import numpy as np
from sklearn.model_selection import train_test_split

data = np.loadtxt('x06Simple.csv', delimiter=',', skiprows=1, usecols=(1, 2, 3))
bias = np.ones((data.shape[0], 1))
data = np.hstack((bias, data))

x = data[:, :-1]
y = data[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

mean = np.mean(x_train[:, 1:], axis=0)
std = np.std(x_train[:, 1:], axis=0, ddof=1)
x_train[:, 1:] = (x_train[:, 1:] - mean) / std
x_test[:, 1:] = (x_test[:, 1:] - mean) / std

k = 1
y_pred = []
for x_query in x_test:
    l1_distance = np.sum(np.abs(x_query - x_train), axis=1)
    weights = np.exp(-l1_distance / (k ** 2))
    W = np.diag(weights)
    x_transpose = np.transpose(x_train)
    XtWX = np.linalg.inv(np.matmul(x_transpose, np.matmul(W, x_train)))
    XtWY = np.matmul(x_transpose, np.matmul(W, y_train))
    theta = np.matmul(XtWX, XtWY)
    y_hat = np.dot(x_query, theta)
    y_pred.append(y_hat)

y_pred = np.array(y_pred)
RMSE = np.sqrt(np.mean((y_test - y_pred) ** 2))
print('RMSE:', RMSE)
