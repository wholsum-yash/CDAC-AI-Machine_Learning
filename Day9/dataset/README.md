# Hierarchical Sample Dataset

This dataset contains structured multi-dimensional numerical features engineered for hierarchical agglomerative clustering, dendrogram tree cuts, and linkage comparison analysis.

---

##  Data Dictionary

| Column Name | Data Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `sample_id` | Integer | ID | Unique identifier for each sample observation. |
| `feature_1` | Float | Continuous | Processed signal parameter 1 (normalized range ~10–90). |
| `feature_2` | Float | Continuous | Processed signal parameter 2 (normalized range ~5–80). |
| `feature_3` | Float | Continuous | Operational intensity indicator (normalized range ~100–500). |
| `feature_4` | Float | Continuous | Frequency density measurement (normalized range ~1.0–10.0). |

---

##  Primary Use Cases
1. **Dendrogram Cut Analysis**: Inspect hierarchical tree height gaps to identify distinct cluster splits.
2. **Linkage Criterion Benchmark**: Compare cluster formation structures across `ward`, `complete`, `average`, and `single` linkage methods.
3. **Cophenetic Validation**: Compute cophenetic correlation coefficients across different distance metrics.