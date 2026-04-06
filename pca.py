# Principal Component Analysis (PCA) - From Scratch
# Dataset: LFW People (Labeled Faces in the Wild) via sklearn.datasets
# Dependencies: numpy, matplotlib, sklearn (fetch_lfw_people, train_test_split only)
#
# Implementation: Computes PCA manually by building the covariance matrix X^T X / N,
# then extracting eigenvalues and eigenvectors using numpy. Eigenvectors are sorted
# by descending eigenvalue to find the principal components. Supports:
#   - Projecting data onto top-k components
#   - Choosing k automatically based on a variance explained threshold (default 95%)
#   - Reconstructing images from projections
#   - Visualizing faces that score highest/lowest on each principal component
#
# Output: Plots of max/min PC1 and PC2 faces, most important eigenvector visualized
#         as a face, reconstructed images using 1 and k components,
#         and the number of components needed to explain 95% of variance

import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

mask = np.zeros(people.target.shape, dtype=bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)


def compute_cov_val_vec(x):
    cov = (x.T @ x) / x.shape[0]
    val, vec = np.linalg.eig(cov)
    return cov, val, vec


def sort_eigenval(val):
    sorted_indices = np.argsort(val)[::-1]
    return sorted_indices, val[sorted_indices]


def top_k_eigenvec(sorted_indices, vec, k):
    return vec[:, sorted_indices[:k]]


def compute_z(x, top_eigenvec):
    return x @ top_eigenvec


def choose_k(eigenvals, threshold):
    total_variance = np.sum(eigenvals)
    total = 0
    for k, val in enumerate(eigenvals):
        total += val
        if total / total_variance >= threshold:
            return k + 1
    return len(eigenvals)


def plot_face(image_vector, title, cmap):
    plt.imshow(image_vector.reshape(image_shape), cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0, ddof=1)
x_train_std = (X_train - mean) / std

cov, val, vec = compute_cov_val_vec(x_train_std)
sorted_eigval_indices, sorted_eigval = sort_eigenval(val)
top_eigvec = top_k_eigenvec(sorted_eigval_indices, vec, 2)
z_train = compute_z(x_train_std, top_eigvec)

plot_face(X_train[np.argmax(z_train[:, 0])], "Max PC1", 'gray')
plot_face(X_train[np.argmin(z_train[:, 0])], "Min PC1", 'gray')
plot_face(X_train[np.argmax(z_train[:, 1])], "Max PC2", 'gray')
plot_face(X_train[np.argmin(z_train[:, 1])], "Min PC2", 'gray')

primary_pc = top_eigvec[:, 0]
plot_face(primary_pc, "Most Important Principal Component", 'gray_r')

x0_std = x_train_std[0, :]
projection = x0_std @ primary_pc
reconstructed = (projection * primary_pc) * std + mean
plot_face(reconstructed, "Reconstruction using primary component", 'gray')

k = choose_k(sorted_eigval, threshold=0.95)
print("Components needed for 95% variance:", k)

top_eigvec_k = top_k_eigenvec(sorted_eigval_indices, vec, k)
z0 = x0_std @ top_eigvec_k
reconstructed_k = (z0 @ top_eigvec_k.T) * std + mean
plot_face(reconstructed_k, f"Reconstruction using {k} components", 'gray')

plot_face(x0_std * std + mean, "Original image", 'gray')
