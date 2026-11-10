# Assignment : Predict Sales Revenue from amount spent on digital ads
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 1. Dataset Generation
np.random.seed(101)
n_campaigns = 40

ad_spend_k = np.random.uniform(5, 50, size=n_campaigns)  # Ad spend in ₹10,000s
revenue_k = 12 + 3.4 * ad_spend_k + np.random.normal(0, 8, size=n_campaigns)  # Revenue in ₹10,000s

df_marketing = pd.DataFrame({
    "ad_spend_inr_k": ad_spend_k.round(2),
    "revenue_inr_k": revenue_k.round(2)
})
print(df_marketing.head())
print("=" * 60)