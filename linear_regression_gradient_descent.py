# Linear Regression - Gradient Descent
# Dataset: x06Simple.csv (columns: x1, x2, y)
# Dependencies: numpy, matplotlib
#
# Implementation: Iteratively updates theta by computing the gradient of MSE loss
# and stepping in the direction that reduces error. Stops when RMSE percent change
# drops below 2^-23 or after 1000 iterations. Standardizes features using training mean/std.
#
# Output: Final model equation, test RMSE, and a plot of train/test RMSE over iterations

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

np.random.seed(0)
theta = np.random.uniform(-1, 1, x_train.shape[1])

learning_rate = 0.01
min_percent_change = 2 ** -23

prev_rmse = 1
i = 0
train_rmse_array = []
test_rmse_array = []

while True:
    error = np.matmul(x_train, theta) - y_train
    x_transpose = np.transpose(x_train)
    theta = theta - (learning_rate / len(x_train) * np.matmul(x_transpose, error))

    y_train_predicted = np.matmul(x_train, theta)
    y_test_predicted = np.matmul(x_test, theta)

    train_rmse = np.sqrt(np.mean((y_train - y_train_predicted) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - y_test_predicted) ** 2))
    train_rmse_array.append(train_rmse)
    test_rmse_array.append(test_rmse)

    change = abs((prev_rmse - train_rmse) / prev_rmse)
    prev_rmse = train_rmse

    if i >= 1000 or change < min_percent_change:
        break
    i += 1

print(f"Final model: y = {theta[0]:.4f} + {theta[1]:.4f}x1 + {theta[2]:.4f}x2")
print("Final Testing RMSE:", test_rmse)

plt.plot(train_rmse_array, label="Train RMSE")
plt.plot(test_rmse_array, label="Test RMSE")
plt.xlabel("Iteration number")
plt.ylabel("RMSE")
plt.legend()
plt.show()
