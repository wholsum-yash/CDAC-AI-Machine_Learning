"""
Title: Healthcare Patient Analytics Lifecycle
Description: Fill in the TODO sections across the 5 lifecycle phases using
             'patient_health_data.csv'.

"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, LinearRegression from sklearn.linear_model, and evaluation metrics
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup
    - Goal: Load 'patient_health_data.csv' and inspect structure.
    - Check for null values and print summary info.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    df = pd.read_csv("patient_health_data.csv")

    # TODO: Print first 5 rows and verify dataset shape
    print(f"Shape: {df.shape}")
    print("Head: \n", df.head)
    print("-" * 65)
    # TODO: Check for any missing values
    print("Missing Values:\n ", df.isna().sum())
    print("-" * 65)

    return df


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Data Measures
    - Goal: Compute measures of central tendency and dispersion
      for 'Systolic_BP'.
    - Check for potential outliers using the IQR method.
    """
    print("\n[PHASE 2] Descriptive Statistics ")
    print("-" * 65)
    # TODO: Calculate Mean, Median, Std Dev, Variance, and IQR for Systolic_BP
    print("Mean: \n", df.mean(numeric_only=True))
    print("-" * 65)
    print("Median: \n", df.median(numeric_only=True))
    print("-" * 65)
    print("Std: \n", df.std(numeric_only=True))
    print("-" * 65)
    print("Variance: \n", df.var(numeric_only=True))
    print("-" * 65)

    q1 = df["Systolic_BP"].quantile(0.25)
    q3 = df["Systolic_BP"].quantile(0.75)

    iqr = q3 - q1
    print("IQR: ", iqr)
    print("-" * 65)

    # TODO: Identify any values falling outside 1.5 * IQR bounds
    lower = q1 - 1.5 * iqr
    upper = q3 - 1.5 * iqr

    df["iqr_outliers"] = (df["Systolic_BP"] < lower) | (df["Systolic_BP"] > upper)
    print("Outliers: ", df["iqr_outliers"].sum())
    print("-" * 65)
    return df


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Association & Correlation
    - Goal: Examine relationships between BMI, Daily Steps, and Systolic BP.
    - Compute Pearson correlation matrix.
    """
    print("\n[PHASE 3] Bivariate Association ")
    print("-" * 65)
    # TODO: Select numeric columns and compute Pearson correlation matrix
    df_num = df.select_dtypes(include=np.number)
    pearson_corr = df_num.corr(method="pearson")
    print("Pearson Corr: ", pearson_corr)
    print("-" * 65)

    # TODO: Highlight correlations involving Systolic_BP
    for col in df_num:
        if col != "Systolic_BP":
            corr = df["Systolic_BP"].corr(df[col])
            df_num["corr"] = corr
        print(f"Correlation:\n{col} and Systolic_BP \n {df_num['corr']}")
    return df


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling
    - Goal: Predict 'Systolic_BP' based on 'BMI'.
    - Train a Simple Linear Regression model.
    """
    print("\n[PHASE 4] Predictive Modeling ")
    print("-" * 65)

    # TODO: Set X = df[['BMI']] and y = df['Systolic_BP']
    X = df[["BMI"]]
    y = df["Systolic_BP"]

    # TODO: Fit Linear Regression model and output slope & intercept
    model = LinearRegression()
    trained_model = model.fit(X, y)

    intercept = model.intercept_
    slope = model.coef_

    print("Intercept: ", intercept)
    print("-" * 65)
    print("Slope: ", slope)
    print("-" * 65)
    return trained_model, X, y


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & Prescriptive Recommendations (5 Marks)
    - Goal: Evaluate model performance using R-squared and MAE.
    - Provide prescriptive clinical recommendations based on findings.
    """
    print("\n[PHASE 5] Evaluation & Clinical Action (5 Marks)")
    print("-" * 65)
    # TODO: Calculate R2 score and Mean Absolute Error (MAE)
    y_predict = model.predict(X)
    r2 = r2_score(y, y_predict)
    mae = mean_absolute_error(X, y_predict)

    print("R2 Score: ", r2)
    print("-" * 65)
    print("MAE: ", mae)
    print("-" * 65)
    # TODO: Print prescriptive clinical action items

    if r2 > 0.80:
        print("Model performs good r2 above 0.8")

    elif r2 < 0.80 and r2 > 0.50:
        print("Model is performing average r2 between 0.79 and 0.5")
    else:
        print("Model is perform below standard r2 is less than 0.5")


def run_analytics_lifecycle():
    """
    Facade Orchestrator function.
    """
    print("=" * 65)
    print("STARTING HEALTHCARE ANALYTICS LIFECYCLE ")
    print("=" * 65)

    # Executing phases chronologically
    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X, y = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()
