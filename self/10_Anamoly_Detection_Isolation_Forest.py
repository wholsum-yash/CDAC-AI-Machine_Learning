# ---------- Q10: Anomaly detection with Isolation Forest ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, accuracy_score)

# 1. Inliers from 3 clusters + 20 clearly separated outliers
X_in, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
rng = np.random.RandomState(42)
X_out = rng.uniform(low=-12, high=12, size=(20, 2))

X = np.vstack([X_in, X_out])
true_labels = np.array([0] * len(X_in) + [1] * len(X_out))  # 0 = inlier, 1 = outlier
print("Outlier proportion:", round(true_labels.mean(), 3))

# 2. Plot the generated data
plt.figure()
plt.scatter(X[true_labels == 0, 0], X[true_labels == 0, 1],
            c="steelblue", s=20, label="inliers")
plt.scatter(X[true_labels == 1, 0], X[true_labels == 1, 1],
            c="red", marker="x", s=60, label="true outliers")
plt.legend(); plt.title("Synthetic data with injected outliers")
plt.show()

# 3-4. Fit Isolation Forest with two contamination settings and compare
for cont in [0.05, 0.10]:
    iso = IsolationForest(contamination=cont, random_state=42)
    pred = iso.fit_predict(X)                 # +1 = inlier, -1 = outlier
    pred_labels = (pred == -1).astype(int)    # convert to 0/1 to match truth

    # 5. Metrics
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    print(f"\n===== contamination = {cont} =====")
    print("Confusion matrix [[TN, FP], [FN, TP]]:\n", cm)
    print("Precision:", round(precision_score(true_labels, pred_labels, zero_division=0), 3),
          "| Recall:", round(recall_score(true_labels, pred_labels, zero_division=0), 3),
          "| F1:", round(f1_score(true_labels, pred_labels, zero_division=0), 3),
          "| Accuracy:", round(accuracy_score(true_labels, pred_labels), 3))

    # 6. Plot detected anomalies vs missed outliers
    plt.figure()
    plt.scatter(X[pred_labels == 0, 0], X[pred_labels == 0, 1],
                c="lightgray", s=20, label="predicted inlier")
    plt.scatter(X[pred_labels == 1, 0], X[pred_labels == 1, 1],
                c="red", marker="x", s=60, label="predicted outlier")
    plt.legend(); plt.title(f"Isolation Forest (contamination={cont})")
    plt.show()
