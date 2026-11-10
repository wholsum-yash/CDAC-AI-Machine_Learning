"""
Title: E-Commerce Customer Analytics Lifecycle 
Description: Fill in the TODO sections across the 5 lifecycle phases using 
             'ecommerce_customer_data.csv'.

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
    - Goal: Load 'ecommerce_customer_data.csv' and inspect structure.
    - Check for null values and print summary info.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    df = pd.read_csv("ecommerce_customer_data.csv")
    # TODO: Print first 5 rows and verify dataset shape
    print("df_head: \n",df.head(5))
    # TODO: Check for any missing values
    print("Missing Values: \n", df.isna().sum())
    return df

def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Data Measures 
    - Goal: Compute measures of central tendency and dispersion 
      for 'Annual_Spend_USD'.
    - Check for potential outliers using the IQR method.
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    
    # TODO: Calculate Mean, Median, Std Dev, Variance, and IQR for Annual_Spend_USD
    print("Mean: ",df['Annual_Spend_USD'].mean(numeric_only = True))
    print("-" * 65)
    print("Median: ",df['Annual_Spend_USD'].median(numeric_only = True))
    print("-" * 65)
    print("Std: ",df['Annual_Spend_USD'].std(numeric_only = True))
    print("-" * 65)
    print("Variance: ",df['Annual_Spend_USD'].var(numeric_only = True))
    print("-" * 65)

    # TODO: Identify any values falling outside 1.5 * IQR bounds
    q1 = df['Annual_Spend_USD'].quantile(0.25)
    q3 = df['Annual_Spend_USD'].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5*iqr
    upper = q3 - 1.5*iqr

    outliers = df[(df['Annual_Spend_USD'] < lower) | (df['Annual_Spend_USD'] > upper)]
    print("Outliers: \n", outliers)
    print("-" * 65)
    return df

def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Association & Correlation 
    - Goal: Examine relationships between Browsing Minutes, Items in Cart, and Annual Spend.
    - Compute Pearson correlation matrix.
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)
    # TODO: Select numeric columns and compute Pearson correlation matrix
    df_numeric = df.select_dtypes(include = np.number)
    pearson_corr = df_numeric.corr(method = "pearson")
    # TODO: Highlight correlations involving Annual_Spend_USD
    print("Corr: \n",pearson_corr[:len(pearson_corr)-1]['Annual_Spend_USD'])
    print("-" * 65)
    return df

def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling 
    - Goal: Predict 'Annual_Spend_USD' based on 'Browsing_Minutes'.
    - Train a Simple Linear Regression model.
    """
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    # TODO: Set X = df[['Browsing_Minutes']] and y = df['Annual_Spend_USD']
    X = df[['Browsing_Minutes']]
    y = df['Annual_Spend_USD']
    # TODO: Fit Linear Regression model and output slope & intercept
    model = LinearRegression()
    trained_model = model.fit(X,y)
    intercept = trained_model.intercept_
    slope = trained_model.coef_

    print("Intercept: ", intercept)
    print("-" * 65)
    print("Slope: ", slope)
    print("-" * 65)
    return trained_model, X, y

def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & Prescriptive Recommendations 
    - Goal: Evaluate model performance using R-squared and MAE.
    - Provide prescriptive business recommendations based on findings.
    """
    print("\n[PHASE 5] Evaluation & Business Action")
    print("-" * 65)
    # TODO: Calculate R2 score and Mean Absolute Error (MAE)
    y_predict = model.predict(X)
    r2 = r2_score(y, y_predict)
    mae = mean_absolute_error(y, y_predict)
    # TODO: Print prescriptive marketing action items

    if r2 > 0.80:
        print("Model is Good, r2 > 0.8")
    elif r2 < 0.80 and r2 > 0.50:
        print("Model is Average, r2 < 0.8 but r2 > 0.50")
    else:
        print("Model is bad, r2 < 0.50")

def run_analytics_lifecycle():
    """
    Facade Orchestrator function.
    """
    print("=" * 65)
    print("STARTING E-COMMERCE ANALYTICS LIFECYCLE ")
    print("=" * 65)
    
    # Executing phases chronologically
    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X, y = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()
