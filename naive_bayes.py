# Naive Bayes - Gaussian
# Dataset: spambase.data (57 features + 1 binary spam label)
# Dependencies: numpy, sklearn (train_test_split only)
#
# Implementation: Computes class priors and per-feature Gaussian likelihoods
# (mean and variance) from the training set separately for spam and non-spam.
# At inference time, computes log-probabilities for each class using the
# Gaussian log-likelihood formula and assigns the class with the higher score.
# A small epsilon (1e-9) is added to variances to avoid log(0) errors.
#
# Output: Precision, Recall, F1-score, and Accuracy on test set

import numpy as np
from sklearn.model_selection import train_test_split

data = np.loadtxt('spambase.data', delimiter=',')

x = data[:, :-1]
y = data[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0, shuffle=True)

mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0, ddof=1)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

spam_data = x_train[y_train == 1]
nonspam_data = x_train[y_train == 0]
prior_spam = spam_data.shape[0] / x_train.shape[0]
prior_nonspam = nonspam_data.shape[0] / x_train.shape[0]

mean_spam = np.mean(spam_data, axis=0)
var_spam = np.var(spam_data, axis=0, ddof=1)
mean_nonspam = np.mean(nonspam_data, axis=0)
var_nonspam = np.var(nonspam_data, axis=0, ddof=1)

eps = 1e-9
pred = []
for sample in x_test:
    log_prob_spam = np.log(prior_spam)
    log_prob_nonspam = np.log(prior_nonspam)

    for j in range(sample.shape[0]):
        log_prob_spam += -0.5 * np.log(2 * np.pi * var_spam[j] + eps) - ((sample[j] - mean_spam[j]) ** 2) / (2 * var_spam[j] + eps)
        log_prob_nonspam += -0.5 * np.log(2 * np.pi * var_nonspam[j] + eps) - ((sample[j] - mean_nonspam[j]) ** 2) / (2 * var_nonspam[j] + eps)

    pred.append(1 if log_prob_spam > log_prob_nonspam else 0)

y_test_pred = np.array(pred)

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
