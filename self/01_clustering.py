# ---------- Q1: Clustering on the Iris dataset ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Load data. Keep species labels aside (used only at the very end).
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y_true = iris.target

# 2. Check missing values, then standardize (clustering is distance-based).
print("Missing values per column:\n", X.isna().sum())
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# 3. Elbow curve: K-Means for k = 1..10
wss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    wss.append(km.inertia_)

plt.figure()
plt.plot(range(1, 11), wss, marker='o')
plt.xlabel("k (number of clusters)")
plt.ylabel("WSS (inertia)")
plt.title("Elbow curve")
plt.show()

# 4. Silhouette scores for candidate k values
print("\nSilhouette scores:")
for k in range(2, 7):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs)
    print(f"k={k}  silhouette={silhouette_score(Xs, labels):.3f}")

# 5. Choose k=3 (elbow + silhouette) and profile the clusters
k = 3
km = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_km = km.fit_predict(Xs)

X_profile = X.copy()
X_profile["cluster"] = labels_km
print("\nCluster sizes:\n", X_profile["cluster"].value_counts().sort_index())
print("\nCluster means (original units):\n", X_profile.groupby("cluster").mean().round(2))

# 6. Agglomerative (hierarchical) clustering + dendrogram
labels_hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xs)
print("\nHierarchical vs K-Means agreement (ARI):",
      round(adjusted_rand_score(labels_km, labels_hc), 3))

plt.figure(figsize=(10, 4))
dendrogram(linkage(Xs, method="ward"), truncate_mode="lastp", p=15)
plt.title("Dendrogram (Ward linkage)")
plt.xlabel("Cluster size / sample index")
plt.show()

# 7. DBSCAN with two settings; report noise counts
for eps, ms in [(0.5, 5), (0.8, 5)]:
    db = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xs)
    n_noise = np.sum(db == -1)
    n_clusters = len(set(db)) - (1 if -1 in db else 0)
    print(f"DBSCAN eps={eps}, min_samples={ms} -> clusters={n_clusters}, noise={n_noise}")

# 8. PCA 2-D plot coloured by K-Means clusters
Xp = PCA(n_components=2, random_state=42).fit_transform(Xs)
plt.figure()
plt.scatter(Xp[:, 0], Xp[:, 1], c=labels_km, cmap="viridis", s=40)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("K-Means clusters in PCA space")
plt.show()

# Post-hoc check only (labels were NOT used for training)
print("K-Means vs true species ARI:", round(adjusted_rand_score(y_true, labels_km), 3))
