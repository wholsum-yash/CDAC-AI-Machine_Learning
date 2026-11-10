# ---------- Q2: Regression model bake-off ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
print(X.describe().round(2))
print("Target stats:", y.min(), y.mean().round(2), y.max())

# 2. Train/test split (test set kept untouched until the end)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. Pipelines (scaling + polynomial features INSIDE the pipeline = no leakage)
def scaled(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])

models = {
    "Linear": (scaled(LinearRegression()),
               {"model__fit_intercept": [True, False]}),
    "Ridge": (scaled(Ridge()),
              {"model__alpha": [0.01, 0.1, 1, 10, 100]}),
    "Lasso": (scaled(Lasso(max_iter=10000)),
              {"model__alpha": [0.001, 0.01, 0.1, 1, 10]}),
    "ElasticNet": (scaled(ElasticNet(max_iter=10000)),
                   {"model__alpha": [0.001, 0.01, 0.1, 1],
                    "model__l1_ratio": [0.2, 0.5, 0.8]}),
    "Poly2": (Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                        ("scaler", StandardScaler()),
                        ("model", LinearRegression())]),
              {"model__fit_intercept": [True, False]}),
}

# 4-5. Tune on training folds, then evaluate ONCE on the test set
results, fitted = [], {}
for name, (pipe, grid) in models.items():
    gs = GridSearchCV(pipe, grid, cv=5, scoring="neg_mean_squared_error")
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    fitted[name] = best
    pred = best.predict(X_test)
    results.append({
        "Model": name,
        "Best params": gs.best_params_,
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    })

results_df = pd.DataFrame(results).set_index("Model").round(3)
print("\nModel comparison:\n", results_df)

# 6. Actual vs predicted plot for the best model (lowest RMSE)
best_name = results_df["RMSE"].idxmin()
pred = fitted[best_name].predict(X_test)
plt.figure()
plt.scatter(y_test, pred, alpha=0.7)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted ({best_name})")
plt.show()

# Coefficients of linear and regularized models
for name in ["Linear", "Ridge", "Lasso"]:
    coefs = pd.Series(fitted[name].named_steps["model"].coef_, index=X.columns)
    print(f"\n{name} coefficients:\n", coefs.round(2))
