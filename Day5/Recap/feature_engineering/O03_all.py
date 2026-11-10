'''
1. **Zero-Variance Feature Removal**
* Dropping features that contain only a single unique value across the entire dataset, providing zero predictive variance.


2. **Quasi-Constant Feature Removal**
* Eliminating features where a vast majority of rows (e.g., over 99%) share the exact same single value, offering little to no discriminative signal.


3. **Highly Correlated & Redundant Feature Removal**
* Removing multicollinearity by dropping one of a pair of features exhibiting high correlation coefficients or high Variance Inflation Factors (VIF).


4. **Missing Value Imputation & Missingness Indicators**
* Applying statistical or advanced imputation (mean, median, mode, KNN, MICE) and creating binary indicator columns (`is_missing`) where the absence of data itself holds predictive weight.


5. **Outlier Treatment**
* Capping, flooring, or Winsorizing extreme continuous values to prevent outliers from distorting distance-based or parametric models.


6. **Categorical Encoding & Rare Category Handling**
* Transforming categorical variables (One-Hot, Ordinal) while explicitly grouping infrequent long-tail categories into an "Other" bucket to prevent unseen category errors.


7. **High-Cardinality Feature Management**
* Managing high-cardinality columns (such as zip codes or user IDs) using smoothed Target Encoding or Feature Hashing to avoid dimensional explosion.


8. **Mathematical & Power Transformations**
* Applying Log, Square Root, Box-Cox, or Yeo-Johnson transformations to stabilize variance and normalize highly skewed continuous distributions.


9. **Feature Scaling & Normalization**
* Applying Standardization (Z-score), Min-Max scaling, or Robust Scaling to ensure all numeric features lie on comparable scales.


10. **Date-Time & Temporal Decomposition**
* Extracting calendar/time components (year, month, day, hour, day of week, elapsed time) and applying cyclical Sine/Cosine transformations for periodic features.


11. **Binning & Discretization**
* Converting continuous variables into discrete intervals using equal-width, equal-frequency (quantile), or domain-driven thresholds.


12. **Dimensionality Reduction & Automated Feature Selection**
* Selecting optimal feature subsets using filter methods, wrapper methods (RFE), embedded methods (Lasso, tree importances), or projecting high-dimensional space using PCA/LDA.
'''

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# DATASET: Employee Performance (20 records)
# Designed to showcase all 12 feature engineering steps.
# ==========================================
data = {
    "always_zero": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], # 1. Zero-Variance
    "laptop_badge": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], # 2. Quasi-Constant
    "hours_worked": [35, 40, 45, 38, 42, 50, 44, 39, 41, 48, 46, 37, 43, 49, 40, 36, 55, 42, 47, 38],
    "minutes_worked": [2100, 2400, 2700, 2280, 2520, 3000, 2640, 2340, 2460, 2880, 
                       2760, 2220, 2580, 2940, 2400, 2160, 3300, 2520, 2820, 2280], # 3. Highly Correlated with hours_worked
    "salary": [50000, 52000, np.nan, 53000, 60000, 75000, 62000, 51000, np.nan, 80000, 
               85000, 52000, 61000, 78000, 65000, 50000, 180000, 63000, 70000, 49000], # 4. Missing values & 5. Outlier (180000)
    "department": ["IT", "HR", "IT", "Sales", "IT", "HR", "Sales", "IT", "HR", "IT", 
                   "Sales", "IT", "HR", "IT", "Sales", "IT", "QuantumPhysics", "IT", "HR", "Sales"], # 6. Categorical with rare category ('QuantumPhysics')
    "postal_code": ["56001", "56002", "56003", "56004", "56005", "56006", "56007", "56008", "56009", "56010",
                    "56011", "56012", "56013", "56014", "56015", "56016", "56017", "56018", "56019", "56020"], # 7. High-Cardinality
    "bonus_amount": [100, 150, 200, 120, 300, 800, 400, 110, 250, 900, 
                     1000, 1300, 350, 850, 450, 90, 5000, 32050, 500, 105], # 8. Skewed continuous distribution
    "satisfaction_score": [3.2, 4.1, 4.5, 3.8, 4.0, 4.9, 4.2, 3.5, 3.9, 4.8, 
                           4.7, 3.6, 4.1, 4.8, 4.3, 3.1, 5.0, 4.2, 4.6, 3.4], # 11. Continuous for Binning
    "join_date": ["2022-01-10", "2021-05-15", "2023-03-20", "2020-11-05", "2022-07-12", "2019-01-01", "2021-08-19", "2023-02-14", "2022-09-30", "2018-04-11",
                  "2020-06-25", "2023-01-05", "2021-10-10", "2019-12-15", "2022-04-01", "2023-05-01", "2017-06-18", "2021-03-12", "2022-11-22", "2023-04-10"], # 10. Date-Time
    "promoted": [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0] # Target variable
}

df = pd.DataFrame(data)
X = df.drop(columns=["promoted"])
Y = df["promoted"]

print(f"Original Shape: {X.shape}\n" + "-"*50)

# ==========================================
# 1. Zero-Variance Feature Removal
# ==========================================
vt_zero = VarianceThreshold(threshold=0.0)
X_num = X.select_dtypes(include=[np.number])
vt_zero.fit(X_num)
X_num = X_num.loc[:, vt_zero.get_support()]
# print("1. After Zero-Variance Removal (Dropped 'always_zero')")

# ==========================================
# 2. Quasi-Constant Feature Removal
# ==========================================
vt_quasi = VarianceThreshold(threshold=0.04) # Binary threshold p*(1-p) for 95% threshold
vt_quasi.fit(X_num)
X_num = X_num.loc[:, vt_quasi.get_support()]
# print("2. After Quasi-Constant Removal (Dropped 'laptop_badge')")

# ==========================================
# 3. Highly Correlated & Redundant Removal
# ==========================================
corr = X_num.corr().abs()
to_drop = set()
for i in range(len(corr.columns)):
    for j in range(i):
        if corr.iloc[i, j] > 0.90:
            to_drop.add(corr.columns[i])
X_num = X_num.drop(columns=list(to_drop))
# print(f"3. After Redundancy Removal (Dropped correlated: {list(to_drop)})")

# ==========================================
# 4. Missing Value Imputation & Indicators
# ==========================================
X_num["salary_is_missing"] = X_num["salary"].isna().astype(int)
X_num["salary"] = X_num["salary"].fillna(X_num["salary"].median())
# print("4. Missing indicator created & salary imputed with median.")

# ==========================================
# 5. Outlier Treatment (Winsorization)
# ==========================================
cap = X_num["salary"].quantile(0.95)
X_num["salary"] = np.where(X_num["salary"] > cap, cap, X_num["salary"])
# print(f"5. Outliers capped in 'salary' at 95th percentile ({cap}).")

# ==========================================
# 6. Categorical Encoding & Rare Categories
# ==========================================
# X_cat = X.select_dtypes(include=[object]).copy() 
X_cat = X.select_dtypes(include=['object','str']).copy()
# Group rare categories appearing < 2 times into 'Other'
counts = X_cat["department"].value_counts()
X_cat["department"] = X_cat["department"].replace(counts[counts < 2].index, "Other")
X_cat = pd.get_dummies(X_cat, columns=["department"], drop_first=True)
# print("6. Rare categories grouped and One-Hot Encoding applied.")

# ==========================================
# 7. High-Cardinality Feature Management
# ==========================================
# Use Frequency Encoding for high-cardinality postal_code to prevent dimension explosion
X_cat["postal_code_freq"] = X_cat["postal_code"].map(X_cat["postal_code"].value_counts())
X_cat = X_cat.drop(columns=["postal_code"])
# print("7. High-cardinality 'postal_code' converted via Frequency Encoding.")

# drop column join_date from X_cat
X_cat = X_cat.drop(columns=["join_date"])

# ==========================================
# 8. Mathematical & Power Transformations
# ==========================================
pt = PowerTransformer(method="yeo-johnson")
X_num[["bonus_amount"]] = pt.fit_transform(X_num[["bonus_amount"]])
# print("8. Power Transformation applied to skewed 'bonus_amount'.")

# X_num["bonus_recovered"]=pt.inverse_transform(X_num[["bonus_amount1"]])
# ==========================================
# 9. Feature Scaling & Normalization
# ==========================================
scaler = StandardScaler()
# cols_to_scale = ["hours_worked", "salary", "bonus_amount", "satisfaction_score"] 
cols_to_scale = ["hours_worked","salary", "bonus_amount"]
X_num[cols_to_scale] = scaler.fit_transform(X_num[cols_to_scale])
# print("9. Standardization (Z-score) applied to numeric features.")

# ==========================================
# 10. Date-Time & Temporal Decomposition
# ==========================================
dates = pd.to_datetime(X["join_date"])
X_time = pd.DataFrame({
    "join_year": dates.dt.year,
    "join_month": dates.dt.month,
    "join_dayofweek": dates.dt.dayofweek,
    "tenure_days": (pd.Timestamp("2026-01-01") - dates).dt.days
})
# Cyclical sine/cosine transformation for month
X_time["month_sin"] = np.sin(2 * np.pi * X_time["join_month"]/12)
X_time["month_cos"] = np.cos(2 * np.pi * X_time["join_month"]/12)
# print("10. Date-time components & cyclical features extracted.")

# ==========================================
# 11. Binning & Discretization
# ==========================================
# X_num["satisfaction_bin"] = pd.qcut(df["satisfaction_score"], q=3, labels=[0, 1, 2]).astype(int)
X_num["satisfaction_bin"] = pd.qcut(X_num["bonus_amount"], q=3, labels=[0, 1, 2]).astype(int)
# print("11. Continuous satisfaction score converted into quantile bins.")

# Combine all processed features
X_processed = pd.concat([X_num, X_cat, X_time], axis=1)

# ==========================================
# 12. Dimensionality Reduction & Feature Selection
# ==========================================

'''
# Automated Feature Selection (Select top features using ANOVA F-value)
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_processed, y)

# Or Dimensionality Reduction via PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_selected)

print(f"12. Feature Selection & PCA Complete! Final Reduced Shape: {X_pca.shape}")
'''


selector_vt = VarianceThreshold(threshold=0)
X_processed_clean = pd.DataFrame(
    selector_vt.fit_transform(X_processed), 
    columns=X_processed.columns[selector_vt.get_support()]
)

selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_processed_clean, Y)
selected_features = X_processed_clean.columns[selector.get_support()]


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed_clean)
# print(X_processed_clean)


# Print SelectKBest Results (ANOVA F-values)
selected_mask = selector.get_support()
selected_features = X_processed_clean.columns[selected_mask]
feature_scores = selector.scores_[selected_mask]

print("-" * 60)
print("Top 5 Selected Features (via ANOVA F-value):")
for feat, score in zip(selected_features, feature_scores):
    print(f" - {feat}: F-score = {score:.4f}")

print("-" * 60)

explained_var_ratio = pca.explained_variance_ / np.sum(pca.explained_variance_)

print("-" * 60)
print("PCA Explained Variance Ratio per Component:", explained_var_ratio)
print(f"Total Cumulative Variance Captured: {np.sum(explained_var_ratio)*100:.2f}%")
print("PCA transformed shape:", X_pca.shape)
print("=" * 60)
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
print(X_selected_df)
print("=" * 60)

