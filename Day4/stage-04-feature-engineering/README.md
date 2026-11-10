# Stage 4: Feature Engineering, Transformation & Selection

- This module handles 
    - domain feature generation, 
    - categorical encoding (One-Hot, Target Encoding), 
    - numerical transformations (Box-Cox, Scaling), 
    - dimensionality reduction, 
    - multistage feature selection, and 
    - strict data leakage prevention.

## Feature Engineering, Transformation & Selection
- Stage 4 bridges raw, cleaned telemetry data from Stage 3 and production-ready feature matrices for machine learning models. 
- To prevent data leakage and ensure reproducibility, this module enforces strict **fit-transform separation** (fitting transformers exclusively on training folds and applying them to validation/test splits).

## Defination and Use
### **Feature Engineering** is the process of using domain knowledge and statistical transformations to extract, modify, and construct meaningful features from raw data, making patterns more salient and improving machine learning model performance

### Cases To Address
 - Data Leakage (Target Leakage):
    Using information from the target variable or future states during feature creation (e.g., computing global means across the entire dataset before train-test splitting).
 - High-Cardinality Explosions:
    Applying One-Hot Encoding to features with thousands of unique values (e.g., ZIP codes or user IDs), resulting in massive, memory-intensive sparse matrices.
 - Multicollinearity Inflation:
    Generating redundant derived features (e.g., age, birth year, and decades lived) that introduce extreme collinearity and destabilize linear coefficients.
 - Imputation Leakage:
    Calculating imputation metrics (such as mean or median) on the full dataset instead of strictly computing them on training partitions.
 - Outlier Sensitivity:
    Applying scaling or polynomial expansions without handling extreme outliers first, which can compress the variance of normal data points into a narrow range.

#### Core Feature Engineering Methodologies
1. Handling Missing Values:
    - Statistical imputation (Mean, Median, Mode) or advanced methods (KNN imputer, MICE).
    - Creating binary missingness indicator flags (e.g., income_is_missing) to preserve missing data patterns as a signal.
2. Categorical Encoding:
    - One-Hot Encoding: For low-cardinality nominal variables.
    - Ordinal Encoding: For categories with a natural inherent order (e.g., Low < Medium < High).
    - Target / Mean Encoding: Replacing categories with the expected target value, regularized to prevent overfitting.
3. Transformation & Scaling:
    - Standardization (Z-score): Centering features to mean 0 and variance 1.
    - Normalization (Min-Max): Bounding features between 0 and 1.
    - Mathematical Transformations: Applying Log, Square Root, or Box-Cox transforms to normalize right-skewed distributions.
4. Datetime Decomposition:
    - Extracting sub-components such as year, month, day, day-of-week, hour, and binary flags for weekends or holidays.
    - Calculating time-since elapsed metrics (e.g., days since last transaction).
5. Interaction & Aggregation Features:
    - Mathematical Combinations: Creating ratios, products, or differences (e.g., debt-to-income ratio).
    - Group-based Aggregates: Calculating rolling means, sums, or frequencies grouped by categorical entities.

