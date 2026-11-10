#  Customer Segmentation Dataset

This dataset contains synthetic customer transaction and engagement profiles designed for unsupervised partitioning, customer persona profiling, and cluster analysis using K-Means.

---

##  Data Dictionary

| Column Name | Data Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `customer_id` | Integer | ID | Unique identifier for each customer. |
| `age` | Integer | Years | Customer age (18 – 70). |
| `annual_income_k` | Float | $1,000s | Estimated annual income in thousands of USD. |
| `spending_score` | Integer | Index (1–100) | Normalized score based on purchasing behavior and store engagement. |
| `recency_days` | Integer | Days | Days elapsed since the last purchase. |
| `purchase_frequency` | Integer | Count/Year | Total distinct purchase transactions completed over the past 12 months. |

---

##  Primary Use Cases
1. **Feature Scaling Analysis**: Demonstrates why scaling is necessary when features vary across drastically different orders of magnitude (`recency_days` vs `spending_score`).
2. **Elbow & Silhouette Evaluation**: Serves as a target benchmark to test $K \in [2, 8]$ sweep diagnostics.