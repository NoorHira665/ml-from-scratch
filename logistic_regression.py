# Logistic Regression - Gradient Ascent
# Dataset: spambase.data (57 features + 1 binary spam label)
# Dependencies: numpy, sklearn (train_test_split only)
#
# Implementation: Uses gradient ascent to maximize log-likelihood.
# Applies the sigmoid function to compute predicted probabilities,
# then updates theta using: theta = theta + lr * (1/N) * X^T * (y - y_pred)
# Stops when log-loss change drops below 2^-23 or after 1500 iterations.
# Features are standardized using training mean/std.
#
# Output: Precision, Recall, F1-score, and Accuracy on test set

import numpy as np
from sklearn.model_selection import train_test_split

data = np.loadtxt('spambase.data', delimiter=',')
bias = np.ones((data.shape[0], 1))
data = np.hstack((bias, data))

x = data[:, :-1]
y = data[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0, shuffle=True)

mean = np.mean(x_train[:, 1:], axis=0)
std = np.std(x_train[:, 1:], axis=0, ddof=1)
x_train[:, 1:] = (x_train[:, 1:] - mean) / std
x_test[:, 1:] = (x_test[:, 1:] - mean) / std

np.random.seed(0)
theta = np.random.uniform(-1, 1, x_train.shape[1])

learning_rate = 0.01
loss_abs_value_change = 2 ** -23

i = 0
prev_loss = float('-inf')

while True:
    x_theta = np.matmul(x_train, theta)
    y_pred = 1 / (1 + np.exp(-x_theta))

    N = x_train.shape[0]
    error = y_train - y_pred
    gradient = 1/N * np.matmul(np.transpose(x_train), error)
    theta = theta + learning_rate * gradient

    loss = np.mean(y_train * np.log(y_pred + 1e-15) + (1 - y_train) * np.log(1 - y_pred + 1e-15))
    change = abs(loss - prev_loss)

    if i >= 1500 or change < loss_abs_value_change:
        break
    prev_loss = loss
    i += 1

y_test_pred_prob = 1 / (1 + np.exp(-np.matmul(x_test, theta)))
y_test_pred = (y_test_pred_prob >= 0.5).astype(int)

true_pos = np.sum((y_test_pred == 1) & (y_test == 1))
false_pos = np.sum((y_test_pred == 1) & (y_test == 0))
true_neg = np.sum((y_test_pred == 0) & (y_test == 0))
false_neg = np.sum((y_test_pred == 0) & (y_test == 1))

precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos > 0) else 0
recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg > 0) else 0
f_1 = (2 * precision * recall) / (precision + recall) if (precision + recall > 0) else 0
accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f_1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
