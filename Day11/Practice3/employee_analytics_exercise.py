"""
Title: Employee Productivity & Analytics Lifecycle
Description: Fill in the TODO sections across the 5 lifecycle phases.

"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, scikit-learn LinearRegression, and metrics
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup (5 Marks)
    - Goal: Load 'employee_productivity.csv' and inspect structure.
    - Check for null values and print summary info.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)

    # TODO: Load CSV file using pandas
    df = pd.read_csv("employee_productivity.csv")

    # TODO: Print first 5 rows and verify shape
    print("Shape: ", df.shape)
    print("-" * 65)
    print("Head: \n", df.head)
    print("-" * 65)

    # TODO: Check for missing values
    df_missing = df.isna().sum()
    print("Missing Values:\n", df_missing)

    return df


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Data Measures
    - Goal: Compute measures of central tendency and dispersion
      for 'Weekly_Tasks_Completed'.
    - Check for potential outliers using IQR method.
    """
    print("\n[PHASE 2] Descriptive Statistics ")
    print("-" * 65)
    # TODO: Calculate Mean, Median, Std Dev, and IQR for Weekly_Tasks_Completed

    df_mean = df.mean(numeric_only=True)
    df_median = df.median(numeric_only=True)
    df_std = df.std(numeric_only=True)

    q1 = np.percentile(df["Weekly_Tasks_Completed"], 25)
    q3 = np.percentile(df["Weekly_Tasks_Completed"], 75)

    iqr = q3 - q1

    print("Mean: \n", df_mean)
    print("-" * 65)
    print("Median: \n", df_median)
    print("-" * 65)
    print("Std: \n", df_std)
    print("-" * 65)
    print("IQR: \n", iqr)
    print("-" * 65)

    # TODO: Identify any values falling outside 1.5 * IQR bounds

    lower = q1 - 1.5 * iqr
    upper = q3 - 1.5 * iqr

    df["iqr_outlier"] = (df["Weekly_Tasks_Completed"] < lower) | (
        df["Weekly_Tasks_Completed"] > upper
    )

    print("Outliers: \n", df["iqr_outlier"])
    return df


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Association & Correlation
    - Goal: Examine relationships between Training Hours, Experience,
      and Tasks Completed.
    - Compute Pearson correlation matrix.
    """
    print("\n[PHASE 3] Bivariate Association ")
    print("-" * 65)
    # TODO: Select numeric columns and compute Pearson correlation matrix

    df = df.select_dtypes(include=np.number)
    df_pearson = df["Training_Hours"].corr(df["Years_Experience"])
    print("Pearson: \n", df_pearson)
    print("-" * 65)
    return df

    # TODO: Highlight the correlation between Training_Hours and Weekly_Tasks_Completed

    plt.figure(figsize=(8, 5))
    sns.scatterplot(df, x="Training_Hours", y="Weekly_Tasks_Completed")
    plt.title("Relation between Training Hours VS Weekly Tasks Completetd")
    plt.xlabel("Training Hours")
    plt.ylabel("Weekly Tasks Completed")
    plt.show()


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling
    - Goal: Predict 'Weekly_Tasks_Completed' based on 'Training_Hours'.
    - Train a Simple Linear Regression model.
    """
    print("\n[PHASE 4] Predictive Modeling ")
    print("-" * 65)
    # TODO: Set X = df[['Training_Hours']] and y = df['Weekly_Tasks_Completed']

    X = df[["Training_Hours"]]
    Y = df[["Weekly_Tasks_Completed"]]

    model = LinearRegression()
    trained_model = model.fit(X, Y)

    # TODO: Fit Linear Regression model and output slope & intercept
    intercept = trained_model.intercept_
    slope = trained_model.coef_

    print("intercept: \n", intercept)
    print("-" * 65)
    print("slope: \n", slope)
    print("-" * 65)
    return trained_model, X, Y


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & Prescriptive Recommendation (5 Marks)
    - Goal: Evaluate model using R-squared and MAE.
    - Provide operational HR recommendations based on data.
    """
    print("\n[PHASE 5] Evaluation & HR Recommendations (5 Marks)")
    print("-" * 65)
    # TODO: Calculate R2 score and Mean Absolute Error (MAE)
    predict = model.predict(X)
    r2 = r2_score(y, predict)
    mae = mean_absolute_error(y, predict)

    # TODO: Print prescriptive action item for employee training investments
    print("R2 Score", r2)
    print("-" * 65)
    print("MAE", mae)


def run_analytics_lifecycle():
    """
    Facade Orchestrator function.
    """
    print("=" * 65)
    print("STARTING EMPLOYEE ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)

    # Executing phases chronologically
    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X, y = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()

