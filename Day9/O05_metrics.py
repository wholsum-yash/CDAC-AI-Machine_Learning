'''
Clustering metrics for evaluating clustering models
1. Inertia metric for evaluating clustering models (also known as elbow method )
    a) works on KMeans and other centroid-based clustering models
    b) lower inertia indicates better clustering
    c) is it intra or inter cluster distance based (intra-cluster distance)
    d) sensitive to scale of features
    pitfalls
        - sensitive to outliers
        - may not work well with non-globular clusters
2. Silhouette score for evaluating clustering models
    a) works on any clustering model
    b) ranges from -1 to 1, higher is better
    c) considers both intra-cluster cohesion and inter-cluster separation
    d) sensitive to the number of clusters
    pitfalls
        - may not work well with clusters of varying density or size
3. Davies-Bouldin index for evaluating clustering models
    a) works on any clustering model
    b) lower is better
    c) considers both intra-cluster cohesion and inter-cluster separation
    pitfalls
        - may not work well with clusters of varying density or size

4. Calinski-Harabasz index for evaluating clustering models
    a) works on any clustering model
    b) higher is better
    c) considers both intra-cluster cohesion and inter-cluster separation
    pitfalls
        - may not work well with clusters of varying density or size
5. Adjusted Rand index for evaluating clustering models
    a) works on any clustering model
    b) ranges from -1 to 1, higher is better
    c) considers both true and predicted cluster assignments
    pitfalls
        - may not work well with clusters of varying size
6. Mutual Information for evaluating clustering models
    a) works on any clustering model
    b) ranges from 0 to 1, higher is better
    c) considers both true and predicted cluster assignments
    pitfalls
        - may not work well with clusters of varying size
7. V-measure for evaluating clustering models
    a) works on any clustering model
    b) ranges from 0 to 1, higher is better
    c) considers both homogeneity and completeness
    pitfalls
        - may not work well with clusters of varying size
8. Homogeneity score for evaluating clustering models
    a) works on any clustering model
    b) ranges from 0 to 1, higher is better
    c) measures if each cluster contains only members of a single class
    pitfalls
        - may not work well with clusters of varying size
9. Completeness score for evaluating clustering models
    a) works on any clustering model
    b) ranges from 0 to 1, higher is better
    c) measures if all members of a given class are assigned to the same cluster
    pitfalls
        - may not work well with clusters of varying size
10. Fowlkes-Mallows index for evaluating clustering models
    a) works on any clustering model
    b) ranges from 0 to 1, higher is better
    c) considers both precision and recall of cluster assignments
    pitfalls
        - may not work well with clusters of varying size
11. Adjusted Mutual Information for evaluating clustering models
    a) works on any clustering model
    b) ranges from 0 to 1, higher is better
    c) considers both true and predicted cluster assignments, adjusted for chance
    pitfalls
        - may not work well with clusters of varying size


'''

'''
Davies-Bouldin Index (DBI)
    a) Popular internal evaluation metric used to validate clustering algorithms, including hierarchical clustering.
    b) It measures how well-separated and compact the resulting clusters are.
    c) Helps determine the optimal cut-off point on a dendrogram. [
    meaning it tells you the ideal number of clusters K by finding the balance where clusters are tightly grouped internally and widely separated from one another.
    ]

'''

'''
How DBI Works:
    1. Intra-cluster distance (si)
        - Measures the average distance between each point in a cluster and the cluster centroid.
    2. Inter-cluster distance (d(ci, cj))
        - Measures the distance between the centroids of two different clusters ci and cj.
    3. Similarity ratio (Rij)
        - Measures the ratio of the sum of intra-cluster distances to the inter-cluster distance between clusters ci and cj.
        - The Davies-Bouldin Index is then computed as the average of the maximum similarity ratios for each cluster.
        Rij = (si + sj) / d(ci, cj)
    4. Cluster Similarity (Ri)
        - For a given cluster i, we find the maximum similarity it shares with any other cluster j.
            Ri = max(Rij) for all j ≠ i
    5. Overall Davies-Bouldin Index (DBI)
        - The average of maximum similarity ratios (Ri) for all K clusters.

        Note: 
        - A lower DBI indicates better clustering, as it implies that clusters are compact and well-separated.
'''

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score

'''
Generate synthetic dataset using make_blobs
make_blobs : Generates isotropic Gaussian blobs for clustering.
             :generates mock 2D data points that naturaly form 4 distinct clusters.
AgglomerativeClustering : A hierarchical clustering method that merges clusters iteratively based on a linkage criterion.
                        :  We loop through it testing k = 2 up to k = 6.
davies_bouldin_score : Computes the Davies-Bouldin Index for evaluating clustering quality.
                    : Measures cluster overlap and dispersion
                    : Lower values indicate better clustering.
                    : which k yields the minimum DBI value.
'''
# step 1 Create a simple Dataset with 4 natural groups (clusters)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
'''
Parameters for make_blobs:
    - n_samples: Total number of data points to generate.
    - centers: Number of distinct clusters to generate.
    - cluster_std: Standard deviation of the clusters.
    - random_state: Seed for reproducibility.
'''
# print(X[:5])  # Print the first 5 data points to verify the dataset
print("=" * 60)

# 2. Test different numbers of clusters (k from 2 to 6)
k_values = range(2, 7)
best_k = None
best_score = float('inf') # Start with infinity so any score is lower
best_labels = None

for k in k_values:
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)
    score = davies_bouldin_score(X, labels)
    print(f"k={k}, DBI={score}")
    if score < best_score:
        best_score = score
        best_k = k
        best_labels = labels

print("=" * 60)
print(f"Best k: {best_k} with DBI: {best_score}")
'''
Visualize the clustering result for the best k
'''
# 3. Visualize the clusters for the optimal k
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], c=best_labels, cmap='viridis', s=50, edgecolor='k')
# plt.title(f"Hierarchical Clustering (Optimal k = {best_k})\nDavies-Bouldin Score: {best_score:.4f}")
# plt.xlabel("Feature X")
# plt.ylabel("Feature Y")
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.show()


