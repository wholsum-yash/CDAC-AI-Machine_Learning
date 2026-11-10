"""
Title: End-to-End Data Analytics Lifecycle
Description: A modular Python script demonstrating business discovery,
             data preparation, exploration, statistics, and predictive modeling.
"""

# ==========================================
# IMPORTS (Libraries for Data, Stats, & Viz)
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# SUB-STEP FUNCTIONS (The Lifecycle Phases)
# ==========================================

df = pd.read_csv("coffee_shop_data.csv")
df_numeric = df.select_dtypes(include=[np.number])


def phase_1_discovery_and_setup():
    """
    Module 1: Foundations, Business Understanding & Setup
    - Define business objective (e.g., Coffee Shop Revenue Analysis).
    - Load the 25-record real-world CSV dataset.
    - Initial inspection (head, info, missing values).
    """
    print("Executing Phase 1: Discovery & Data Setup...")

    # Business Objective:
    #   Determine which factors (e.g., Marketing Spend, Foot Traffic,
    #   Weather) most influence Daily Revenue, and build a simple
    #   predictive model to support budget decisions.

    # TODO: Load dataset using pandas  -> already loaded globally
    # TODO: Perform basic inspection (shape, data types, missing values)

    print("-" * 60)
    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("-" * 60)
    print("Column Data Types:")
    print(df.dtypes)
    print("-" * 60)
    print("First 5 Records:")
    print(df.head())
    print("-" * 60)
    print("Missing Values per Column:")
    print(df.isna().sum())
    print("-" * 60)
    print("Summary Info:")
    df.info()


def phase_2_descriptive_statistics():
    """
    Module 2: Descriptive Statistics & Data Measures
    - Calculate Central Tendency (Mean, Median, Mode).
    - Calculate Dispersion (Standard Deviation, Variance, IQR, Range).
    - Check Distribution Shape & Outliers.
    """
    print("Executing Phase 2: Descriptive Statistics & Measures...")

    revenue = df["Daily_Revenue_USD"]
    marketing = df["Marketing_Spend_USD"]

    # ---- Central Tendency ----
    print("--- Central Tendency ---")
    print(f"Mean Daily Revenue     : {revenue.mean():.2f}")
    print(f"Median Daily Revenue   : {revenue.median():.2f}")
    print(f"Mode Daily Revenue     : {revenue.mode().tolist()}")
    print("-" * 60)
    print(f"Mean Marketing Spend   : {marketing.mean():.2f}")
    print(f"Median Marketing Spend : {marketing.median():.2f}")
    print(f"Mode Marketing Spend   : {marketing.mode().tolist()}")
    print("-" * 60)

    # ---- Dispersion ----
    print("--- Dispersion ---")
    for name, col in [("Daily Revenue", revenue), ("Marketing Spend", marketing)]:
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = col[(col < lower) | (col > upper)]
        print(f"{name}:")
        print(f"  Std Dev : {col.std():.2f}")
        print(f"  Variance: {col.var():.2f}")
        print(f"  Range   : {col.max() - col.min():.2f}")
        print(f"  IQR     : {iqr:.2f}  (Q1={q1:.2f}, Q3={q3:.2f})")
        print(f"  Outliers (1.5*IQR rule): {outliers.tolist()}")
        print(f"  Skewness: {col.skew():.3f}  (0 = symmetric)")
        print("-" * 60)

    # TODO: Handle outliers (here we flag only; small dataset so we keep them)
    print("Note: With only ~25 records, outliers are flagged but retained "
          "to preserve statistical power.\n")


def phase_3_bivariate_association():
    """
    Module 4: Bivariate & Multivariate Association
    - Analyze relationships between variables (e.g., Marketing Spend vs. Revenue).
    - Compute Covariance and Pearson Correlation Matrix.
    - Generate exploratory visualizations (Scatter plots, Correlation heatmap).
    """
    print("Executing Phase 3: Bivariate Association & Visualization...")

    pearson_coffee_corr = df_numeric.corr(method="pearson")
    kendall_coffee_corr = df_numeric.corr(method="kendall")
    covariance = df_numeric.cov()

    print("Pearson Corr: \n", pearson_coffee_corr)
    print("-" * 60)
    print("Kendall Corr: \n", kendall_coffee_corr)
    print("-" * 60)
    print("Covariance \n", covariance)
    print("-" * 60)

    # ---- Scatter: Marketing Spend vs Daily Revenue ----
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x="Marketing_Spend_USD", y="Daily_Revenue_USD")
    plt.title("Daily Revenue VS Marketing Spend Scatterplot")
    plt.legend(["Daily_Revenue_USD (y) vs Marketing_Spend_USD (x)"],
               loc="upper left")
    plt.tight_layout()
    plt.show()

    # ---- Correlation heatmap ----
    plt.figure(figsize=(5, 3))
    sns.heatmap(
        pearson_coffee_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=1.5
    )
    plt.title("Pearson heatmap")
    plt.tight_layout()
    plt.show()


def phase_4_predictive_modeling():
    """
    Module 6 & 8: Predictive Analytics & Regression Modeling
    - Set up a simple predictive model (e.g., Simple Linear Regression).
    - Train the model on independent variables to predict revenue.
    - Establish a baseline evaluation.
    """
    print("Executing Phase 4: Predictive Modeling...")

    # TODO: Define features (X) and target (y)
    X = df[["Marketing_Spend_USD"]]
    y = df["Daily_Revenue_USD"]

    # TODO: Fit a simple linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # TODO: Extract model coefficients and intercept
    print(f"Model Intercept (β0): {model.intercept_:.2f}")
    print(f"Model Coefficient (β1, Marketing Spend): {model.coef_[0]:.4f}")
    print(f"Equation: Daily_Revenue ≈ {model.intercept_:.2f} "
          f"+ {model.coef_[0]:.4f} * Marketing_Spend")

    # Store model + predictions on the training set for Phase 5
    phase_4_predictive_modeling.model = model
    phase_4_predictive_modeling.y_true = y.values
    phase_4_predictive_modeling.y_pred = model.predict(X)
    print("-" * 60)
    print("Model training complete. Predictions stored for Phase 5.")


def phase_5_evaluation_and_reporting():
    """
    Module 6 & 7: Model Evaluation, Generalization & Business Decision
    - Evaluate model performance (e.g., R-squared, Mean Absolute Error).
    - Discuss generalization and overfitting risks on small data.
    - Translate technical findings into prescriptive business recommendations.
    """
    print("Executing Phase 5: Evaluation & Business Action...")

    model = phase_4_predictive_modeling.model
    y_true = phase_4_predictive_modeling.y_true
    y_pred = phase_4_predictive_modeling.y_pred

    # TODO: Calculate evaluation metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"R-squared (R²) : {r2:.4f}  -> {r2 * 100:.2f}% of variance explained")
    print(f"Mean Absolute Error (MAE) : {mae:.2f} USD")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} USD")
    print("-" * 60)

    # ---- Generalization / overfitting discussion ----
    print("Generalization & Overfitting Notes:")
    print("• Dataset has ~25 rows and 1 feature → extremely small sample.")
    print("• No hold-out set (train/test split) is used here to keep the")
    print("  demo simple; in production, we would use k-fold cross-validation.")
    print("• A single-feature model is low-variance but may underfit; adding")
    print("  features (foot traffic, weather, day-of-week) could improve fit.")
    print("-" * 60)

    # TODO: Print final business recommendations
    slope = model.coef_[0]
    print("Business Recommendations:")
    if slope > 0:
        print(f" 1. Marketing Spend shows a POSITIVE association with revenue "
              f"(slope = {slope:.2f}).")
        print(f" 2. Each additional $1 in marketing is associated with an "
              f"estimated ${slope:.2f} increase in daily revenue.")
    else:
        print(" 1. Marketing Spend shows a NEGATIVE association with revenue — "
              "review campaign effectiveness.")
    print(f" 3. R² = {r2:.2f} → model explains {r2*100:.1f}% of revenue "
          f"variability; invest in richer data for stronger inference.")
    print(" 4. Recommended actions: A/B test marketing increments, collect "
          "foot-traffic and weather data, and validate the model on new "
          "stores before scaling marketing budgets.")
    print()


# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================


def run_analytics_lifecycle():
    """
    Facade function that coordinates the entire data analytics
    lifecycle in chronological order.
    """
    print("=" * 60)
    print("STARTING COMPLETE DATA ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 60)

    phase_1_discovery_and_setup()
    print("-" * 60)

    phase_2_descriptive_statistics()
    print("-" * 60)

    phase_3_bivariate_association()
    print("-" * 60)

    phase_4_predictive_modeling()
    print("-" * 60)

    phase_5_evaluation_and_reporting()

    print("=" * 60)
    print("ANALYTICS LIFECYCLE COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_analytics_lifecycle()
