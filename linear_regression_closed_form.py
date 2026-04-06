# Linear Regression - Closed Form Solution
# Dataset: x06Simple.csv (columns: x1, x2, y)
# Dependencies: numpy
#
# Implementation: Solves for theta directly using the normal equation:
#   theta = (X^T X)^-1 X^T y
# No iterative updates needed. Standardizes features using training mean/std.
#
# Output: Final model equation and RMSE on test set

import math
import numpy as np

data = np.loadtxt('x06Simple.csv', delimiter=',', skiprows=1, usecols=(1, 2, 3))
bias = np.ones((data.shape[0], 1))
data = np.hstack((bias, data))

np.random.seed(0)
shuffled_ind = np.random.permutation(len(data))
data_shuffled = data[shuffled_ind]

split_index = math.ceil(len(data_shuffled) * 2/3)
training_data = data_shuffled[:split_index]
testing_data = data_shuffled[split_index:]

mean = np.mean(training_data[:, 1:3], axis=0)
std = np.std(training_data[:, 1:3], axis=0, ddof=1)
training_data[:, 1:3] = (training_data[:, 1:3] - mean) / std
testing_data[:, 1:3] = (testing_data[:, 1:3] - mean) / std

X_train = training_data[:, :-1]
Y_train = training_data[:, -1]
X_test = testing_data[:, :-1]
Y_test = testing_data[:, -1]

X_transpose = np.transpose(X_train)
XTX = np.matmul(X_transpose, X_train)
XTX_inv = np.linalg.inv(XTX)
XTY = np.matmul(X_transpose, Y_train)
theta = np.matmul(XTX_inv, XTY)

Y_pred = np.matmul(X_test, theta)
rmse = np.sqrt(np.mean((Y_test - Y_pred) ** 2))

print(f"Final model: y = {theta[0]:.4f} + {theta[1]:.4f}x1 + {theta[2]:.4f}x2")
print("RMSE:", rmse)
