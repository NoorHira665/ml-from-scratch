# K-Means Clustering
# Dataset: diabetes.csv (first column: label, remaining: features)
# Dependencies: numpy, matplotlib, sklearn (PCA only for visualization)
#
# Implementation: Initializes k centroids randomly from the data. On each iteration,
# assigns each point to its nearest centroid using Euclidean distance, then updates
# centroids to the mean of assigned points. If input has more than 3 features,
# PCA reduces it to 3D for plotting. Stops when centroid movement is below 2^-23
# or after 300 iterations. Purity is computed as weighted average of majority
# class fraction per cluster.
#
# Output: Scatter plot at iteration 1 and convergence, with purity score displayed

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter


def calc_purity(y_pred, y_acc):
    avg_weighted_purity = 0
    clusters = np.unique(y_pred)
    total_samples = len(y_acc)

    for c in clusters:
        cluster_indices = np.where(y_pred == c)[0]
        cluster_labels = y_acc[cluster_indices].ravel()
        if len(cluster_labels) == 0:
            continue
        majority = max(Counter(cluster_labels).values())
        cluster_purity = majority / len(cluster_labels)
        avg_weighted_purity += (len(cluster_labels) / total_samples) * cluster_purity

    return avg_weighted_purity


def plot(X, centroids, y_pred, Y, iteration_num=0, ax=None):
    d = X.shape[1]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow', 'black']

    if ax is None:
        fig = plt.figure(figsize=(7, 6)) if d == 3 else plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d') if d == 3 else fig.add_subplot(111)

    for index, cluster in enumerate(np.unique(y_pred)):
        color = colors[index]
        points = X[y_pred == cluster]
        ax.scatter(*points.T, marker='x', s=40, linewidths=1.5, c=color)
        ax.scatter(*centroids[cluster], marker='o', s=200, edgecolor='black', linewidths=1, c=color)

    purity = calc_purity(y_pred, Y)
    ax.set_title(f"Iteration {iteration_num} | Purity {purity:.5f}")
    plt.show()


def myK_Means(X, Y, k):
    epsilon = 2 ** -23
    if X.shape[1] > 3:
        pca = PCA(n_components=3, whiten=False, svd_solver='auto')
        X = pca.fit_transform(X)

    num_samples = X.shape[0]
    np.random.seed(0)
    random_indices = np.random.choice(num_samples, size=k, replace=False)
    centroids = X[random_indices, :]

    for iter in range(300):
        old_centroids = centroids.copy()

        distances = np.zeros((num_samples, k))
        for i in range(num_samples):
            for j in range(k):
                distances[i, j] = np.sqrt(np.sum((X[i] - centroids[j]) ** 2))

        y_pred = np.zeros(num_samples, dtype=int)
        for i in range(num_samples):
            y_pred[i] = np.argmin(distances[i])

        if iter == 0:
            plot(X, centroids, y_pred, Y, iter + 1)

        for cluster in range(k):
            if np.any(y_pred == cluster):
                centroids[cluster] = X[y_pred == cluster].mean(axis=0)

        diff = np.sum(np.abs(centroids - old_centroids))
        if diff < epsilon:
            plot(X, centroids, y_pred, Y, iter + 1)
            break


data = np.loadtxt('diabetes.csv', delimiter=',')
X = data[:, 1:]
Y = data[:, :1]

mean = np.mean(X, axis=0)
sd = np.std(X, axis=0, ddof=1)
X_std = (X - mean) / sd

myK_Means(X_std, Y, 2)
