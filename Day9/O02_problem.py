import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist

# 1. Create a DataFrame for the sample data points
data = {
    'Sample_ID': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
    'Feature_A': [1, 1, 1, 4, 4, 4],
    'Feature_B': [2, 4, 0, 2, 4, 0]
}

df = pd.DataFrame(data)
print(df)

# 2. Separate what is required for clustering vs. labeling
features = df[['Feature_A', 'Feature_B']]  # Only numerical features for distance math
sample_labels = df['Sample_ID'].values     # Descriptive labels for the chart

# 3. Compute hierarchical/agglomerative clustering using Ward's method
linked = linkage(features, 'ward')
# --- ADDED: Cophenetic Correlation Coefficient ---
# Calculate original pairwise distances from the feature subset
orig_dists = pdist(features)

# Compute cophenetic correlation coefficient and cophenetic distances
c_coeff, cophenetic_dists = cophenet(linked, orig_dists)
print(f"\nCophenetic Correlation Coefficient: {c_coeff:.4f}")

# 4. Plot the dendrogram with DataFrame labels
plt.figure(figsize=(8, 5))
dendrogram(linked,
           labels=sample_labels,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)

plt.title("Hierarchical Clustering Dendrogram from DataFrame")
plt.xlabel("Sample ID")
plt.ylabel("Euclidean Distance")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()