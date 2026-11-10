# ---------- Q4: KNN vs LDA vs QDA ----------
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 1. Load and inspect
data = load_wine()
X, y = data.data, data.target
print("Class counts:", np.bincount(y))
print(pd.DataFrame(X, columns=data.feature_names).describe().loc[["min", "max"]].round(2))

# 2. Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# 3. KNN pipeline + tuning (neighbours and distance weighting)
knn_pipe = Pipeline([("scaler", StandardScaler()),
                     ("knn", KNeighborsClassifier())])
knn_grid = {"knn__n_neighbors": list(range(1, 16)),
            "knn__weights": ["uniform", "distance"]}
knn_gs = GridSearchCV(knn_pipe, knn_grid, cv=5, scoring="f1_macro")
knn_gs.fit(X_train, y_train)
print("\nBest KNN settings:", knn_gs.best_params_)

# 4. LDA and QDA (with and without scaling)
lda = Pipeline([("scaler", StandardScaler()),
                ("lda", LinearDiscriminantAnalysis())]).fit(X_train, y_train)
qda = Pipeline([("scaler", StandardScaler()),
                ("qda", QuadraticDiscriminantAnalysis())]).fit(X_train, y_train)
qda_raw = QuadraticDiscriminantAnalysis().fit(X_train, y_train)  # unscaled

# 5. Compare on the held-out test set
models = {
    "KNN (tuned)": knn_gs.best_estimator_,
    "LDA (scaled)": lda,
    "QDA (scaled)": qda,
    "QDA (unscaled)": qda_raw,
}
rows = []
for name, m in models.items():
    pred = m.predict(X_test)
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_test, pred),
                 "Macro F1": f1_score(y_test, pred, average="macro")})
    print(f"\n{name} confusion matrix:\n{confusion_matrix(y_test, pred)}")

print("\nComparison:\n", pd.DataFrame(rows).set_index("Model").round(3))
