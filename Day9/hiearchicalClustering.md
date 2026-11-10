### Dendrogram Tree Construction & Cut Analysis
- Construct hierarchical agglomerative trees
- Visualize dendrograms
- Evaluate Cophenetic Correlation
- Cut trees at specific height thresholds

### Key Objectives

1. **Hierarchy Building** : Compute linkage matrices : `scipy.cluster.hierarchy.linkage`
2. **Dendrogram Interpretation**: Plot dendrogram trees to locate horizontal cut thresholds and identify natural cluster groupings. 
3. **Cophenetic Validation**: Calculate the Cophenetic Correlation Coefficient. : `scipy.cluster.hierarchy.cophenet`
4. **Cluster Extraction**: Extract discrete cluster assignments : `scipy.cluster.hierarchy.fcluster` and `sklearn.cluster.AgglomerativeClustering`

---

##  Mathematical Metrics

* **Cophenetic Distance**: The height in the dendrogram where two observations $i$ and $j$ are first merged into a common cluster ($c_{ij}$).
* **Cophenetic Correlation Coefficient ($r$)**: Pearson correlation between original pairwise Euclidean distances $d_{ij}$ and cophenetic distances $c_{ij}$:
  $$r = \frac{\sum_{i<j} (d_{ij} - \bar{d})(c_{ij} - \bar{c})}{\sqrt{\sum_{i<j} (d_{ij} - \bar{d})^2 \sum_{i<j} (c_{ij} - \bar{c})^2}}$$