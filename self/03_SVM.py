# ---------- Q3: SVM classification + hyperparameter tuning ----------
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)
from scipy.stats import loguniform

# 1. Load data + stratified split
data = load_breast_cancer()
X, y = data.data, data.target
print("Class counts:", np.bincount(y), "| 0 = malignant, 1 = benign")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Pipeline: scaler -> SVC (scaling learned only from training folds)
pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(random_state=42))])

def report(name, model):
    pred = model.predict(X_test)
    return {"Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Macro F1": f1_score(y_test, pred, average="macro")}

rows = []

# 3. Baseline SVMs: linear and RBF kernels with default C
for kernel in ["linear", "rbf"]:
    base = Pipeline([("scaler", StandardScaler()),
                     ("svc", SVC(kernel=kernel, random_state=42))])
    base.fit(X_train, y_train)
    rows.append(report(f"Baseline {kernel}", base))

# 4a. GridSearchCV over C and gamma
param_grid = [
    {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10, 100]},
    {"svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10, 100],
     "svc__gamma": ["scale", 0.01, 0.1, 1]},
]
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train, y_train)
print("\nGridSearch best:", grid.best_params_)
rows.append(report("GridSearch SVM", grid.best_estimator_))

# 4b. RandomizedSearchCV with a comparable search space
param_dist = [
    {"svc__kernel": ["linear"], "svc__C": loguniform(1e-2, 1e2)},
    {"svc__kernel": ["rbf"], "svc__C": loguniform(1e-2, 1e2),
     "svc__gamma": loguniform(1e-3, 1e0)},
]
rand = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=5,
                          scoring="f1_macro", random_state=42, n_jobs=-1)
rand.fit(X_train, y_train)
print("RandomSearch best:", rand.best_params_)
rows.append(report("RandomSearch SVM", rand.best_estimator_))

# 6. Comparison table
print("\nComparison:\n", pd.DataFrame(rows).set_index("Model").round(3))

# 7. Confusion matrix + classification report for the tuned (GridSearch) model
best = grid.best_estimator_
pred = best.predict(X_test)
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification report:\n",
      classification_report(y_test, pred, target_names=data.target_names))
