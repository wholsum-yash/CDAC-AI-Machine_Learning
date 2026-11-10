import numpy as np
import pandas as pd

# ==========================================
# DATASET: Predicting Job Hiring (20 records)
# ==========================================
# Designed to intentionally showcase:
# 1. Quasi-constant feature (laptop_owned -> almost all 1s)
# 2. Redundant / highly correlated feature (coding_score & typing_speed)
# 3. Missing values (interview_score has NaNs)
# 4. Outlier (years_experience has an extreme value: 35)
# 5. Categorical with rare categories (preferred_os has 'Linux' only once)
# 6. Skewed distribution (salary_expectation)
# ==========================================

data = {
    "years_experience": [1, 2, 3, 2, 4, 5, 6, 3, 5, 7, 8, 2, 4, 6, 5, 3, 35, 4, 5, 2],
    "coding_score":     [45, 50, 60, 55, 65, 70, 75, 58, 72, 80, 85, 52, 68, 78, 74, 59, 95, 66, 71, 48],
    "typing_speed":     [46, 51, 61, 56, 66, 71, 76, 59, 73, 81, 86, 53, 69, 79, 75, 60, 96, 67, 72, 49], 
    "laptop_owned":     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
    "interview_score":  [70, 65, np.nan, 80, 85, 90, 75, 60, np.nan, 95, 88, 72, 78, 82, 88, 65, 92, 70, 80, 62], 
    "preferred_os":     ["Windows", "Mac", "Windows", "Linux", "Mac", "Windows", "Windows", "Mac", "Windows", "Mac", 
                         "Windows", "Mac", "Windows", "Mac", "Windows", "Windows", "Mac", "Windows", "Mac", "Windows"], 
    "salary_expectation": [30000, 32000, 35000, 33000, 40000, 45000, 50000, 34000, 46000, 55000, 
                           60000, 32000, 41000, 52000, 48000, 35000, 150000, 42000, 47000, 31000], 
    "hired":            [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0] 
}

df = pd.DataFrame(data)
print(f"Original Dataset Shape: {df.shape}")
print("-" * 60)