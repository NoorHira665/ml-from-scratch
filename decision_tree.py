# Decision Tree - ID3 (Information Gain)
# Dataset: spambase.data (57 features + 1 binary spam label)
# Dependencies: numpy, sklearn (train_test_split only)
#
# Implementation: Builds a binary decision tree recursively using the DTL algorithm.
# At each node, selects the feature with the highest information gain computed from
# entropy. Features are binarized using the median of each training feature as threshold.
# Leaf nodes return the majority class. Falls back to parent mode when examples are empty.
#
# Output: Precision, Recall, F1-score, and Accuracy on test set

import numpy as np
from sklearn.model_selection import train_test_split


def returnBestFeature(x, y, features):
    p = np.sum(y == 1)
    n = np.sum(y == 0)
    prob_p = p / (n + p)
    prob_n = 1 - prob_p

    eps = 1e-12
    parent_entropy = -((prob_p + eps) * np.log2(prob_p + eps)) - ((prob_n + eps) * np.log2(prob_n + eps))

    maxIG = float('-inf')
    maxIG_feature = None

    for feature in features:
        average_entropy = 0
        for val in [0, 1]:
            indices = np.where(x[:, feature] == val)
            y_subset = y[indices]
            if len(y_subset) == 0:
                continue
            p_i = np.sum(y_subset == 1)
            n_i = np.sum(y_subset == 0)
            prob_pi = p_i / (p_i + n_i)
            prob_ni = 1 - prob_pi
            entropy_subset = -((prob_pi + eps) * np.log2(prob_pi + eps)) - ((prob_ni + eps) * np.log2(prob_ni + eps))
            average_entropy += ((p_i + n_i) / (p + n)) * entropy_subset

        information_gain = parent_entropy - average_entropy
        if information_gain > maxIG:
            maxIG = information_gain
            maxIG_feature = feature

    return maxIG_feature


def mode(y):
    if len(y) == 0:
        return None
    return 1 if np.mean(y) >= 0.5 else 0


def DTL(x, y, features, default):
    if len(y) == 0:
        return default
    unique_vals_y = np.unique(y)
    if len(unique_vals_y) == 1:
        return int(unique_vals_y[0])
    examples_mode = int(mode(y))
    if len(features) == 0:
        return examples_mode

    bestFeature = returnBestFeature(x, y, features)
    tree = {'feature': bestFeature, 'default': examples_mode, 'branches': {}}

    for val in [0, 1]:
        indices = np.where(x[:, bestFeature] == val)
        remaining = [f for f in features if f != bestFeature]
        subtree = DTL(x[indices], y[indices], remaining, examples_mode)
        tree['branches'][val] = subtree

    return tree


def predictClass(tree, sample):
    if not isinstance(tree, dict):
        return tree
    value = sample[tree['feature']]
    if value in tree['branches']:
        return predictClass(tree['branches'][value], sample)
    return tree['default']


data = np.loadtxt('spambase.data', delimiter=',')

x = data[:, :-1]
y = data[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0, shuffle=True)

mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0, ddof=1)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

thresholds = np.median(x_train, axis=0)
x_train = (x_train >= thresholds).astype(int)
x_test = (x_test >= thresholds).astype(int)

features = list(range(x_train.shape[1]))
tree = DTL(x_train, y_train, features, int(mode(y_train)))

predictions = [predictClass(tree, sample) for sample in x_test]
y_test_pred = np.array(predictions)

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
