# ml-from-scratch

Machine learning algorithms implemented from scratch in Python using only NumPy. No ML libraries used for the core logic.

---

## Algorithms

| Algorithm | Dataset | Output |
|---|---|---|
| Linear Regression (Closed Form) | x06Simple.csv | RMSE |
| Linear Regression (Gradient Descent) | x06Simple.csv | RMSE + loss curve plot |
| Locally Weighted Regression | x06Simple.csv | RMSE |
| Logistic Regression | spambase.data | Precision, Recall, F1, Accuracy |
| Naive Bayes (Gaussian) | spambase.data | Precision, Recall, F1, Accuracy |
| Decision Tree (ID3) | spambase.data | Precision, Recall, F1, Accuracy |
| K-Means Clustering | diabetes.csv | Cluster plot + purity score |
| PCA | LFW People (via sklearn) | Reconstructed face images |

---

## Datasets

| File | Description |
|---|---|
| `x06Simple.csv` | Regression dataset with two features and a continuous target |
| `spambase.data` | UCI Spambase dataset, 57 features, binary spam/not-spam label |
| `diabetes.csv` | Diabetes dataset used for unsupervised clustering |
| LFW People | Labeled Faces in the Wild, fetched automatically via sklearn |

---

## Setup

```bash
pip install numpy matplotlib scikit-learn
```

---

## Usage

Each algorithm is self-contained in its own folder. Run any file directly:

```bash
python linear_regression_closed_form/linear_regression_closed_form.py
python logistic_regression/logistic_regression.py
python kmeans/kmeans.py
```

Place the required CSV files in the same directory as the script before running.

---

## Notes

- sklearn is only used for `train_test_split`, `fetch_lfw_people`, and PCA visualization. All algorithm logic is implemented manually with NumPy.
- Features are standardized using training set mean and std throughout to prevent data leakage.
