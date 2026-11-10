# ---------- Q5: Decision Tree vs Gaussian Naïve Bayes on digits ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

# 1. Load data and look at a couple of samples (8x8 images)
digits = load_digits()
X, y = digits.data, digits.target
print("Shape:", X.shape, "| classes:", np.unique(y))

fig, axes = plt.subplots(1, 4, figsize=(8, 2))
for ax, i in zip(axes, range(4)):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"label={digits.target[i]}")
    ax.axis("off")
plt.show()

# 2. Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# 3. Decision tree with cross-validated tuning
tree_grid = {"max_depth": [3, 5, 7, 10, None],
             "min_samples_leaf": [1, 2, 5]}
tree_gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                       tree_grid, cv=5, scoring="f1_macro")
tree_gs.fit(X_train, y_train)
tree = tree_gs.best_estimator_
print("Best tree settings:", tree_gs.best_params_)

# 4. Gaussian Naïve Bayes on the same split
gnb = GaussianNB().fit(X_train, y_train)

# 5. Compare metrics and confusion matrices
for name, m in [("Decision Tree", tree), ("GaussianNB", gnb)]:
    pred = m.predict(X_test)
    print(f"\n===== {name} =====")
    print("Accuracy:", round(accuracy_score(y_test, pred), 3),
          "| Macro F1:", round(f1_score(y_test, pred, average="macro"), 3))
    print(classification_report(y_test, pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

# 6. Tree feature importance (top 10 pixels)
imp = pd.Series(tree.feature_importances_, index=range(X.shape[1]))
print("\nTop 10 important pixels:\n", imp.sort_values(ascending=False).head(10))

# 7. Look at a few misclassified digits
pred_tree = tree.predict(X_test)
wrong = np.where(pred_tree != y_test)[0][:5]
fig, axes = plt.subplots(1, len(wrong), figsize=(10, 2))
for ax, i in zip(np.atleast_1d(axes), wrong):
    ax.imshow(X_test[i].reshape(8, 8), cmap="gray")
    ax.set_title(f"true={y_test[i]}\npred={pred_tree[i]}")
    ax.axis("off")
plt.show()
