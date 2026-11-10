#  Conceptual Summary: K-Means Clustering

K-Means is a centroid-based, iterative partitioning algorithm that divides an unlabeled dataset into $K$ non-overlapping clusters.

---

##  Core Algorithm Mechanics (Lloyd's Algorithm)

1. **Initialization**: Select $K$ initial centroids (randomly or via K-Means++).
2. **Assignment Step**: Assign each observation to its nearest centroid based on Euclidean distance:
   $$c^{(i)} := \arg\min_j \|x^{(i)} - \mu_j\|^2$$
3. **Update Step**: Recompute the mean position (centroid) of each cluster:
   $$\mu_j := \frac{1}{|C_j|} \sum_{i \in C_j} x^{(i)}$$
4. **Convergence Check**: Repeat steps 2 and 3 until centroids stabilize (no change in positions) or `max_iter` is reached.

---

##  K-Means++ Initialization

Standard random initialization can cause convergence to poor local minima. **K-Means++** solves this by spacing initial centroids far apart:
1. Choose the first centroid $\mu_1$ uniformly at random from the data points.
2. For each remaining point $x$, compute its distance $D(x)$ to the nearest already chosen centroid.
3. Select the next centroid with probability proportional to $D(x)^2$.
4. Repeat until $K$ centroids are chosen.

---

##  Key Limitations
* **Assumes Convexity**: Fails on non-globally spherical, arbitrary shapes or interleaved rings (use DBSCAN instead).
* **Distance Sensitivity**: Scale-variant—features must be standardized beforehand.
* **Sensitivity to Outliers**: Extreme values skew cluster mean calculations significantly.


####  K-Means Clustering Cheat Sheet & Takeaways

###  Core Rules of Thumb
1. **Always Scale First**: Run `StandardScaler()` on continuous features prior to fitting `KMeans`.
2. **Use `init='k-means++'`**: Never rely on raw random initialization in production.
3. **Determine $K$ Jointly**: Combine the **Elbow Method** (Inertia inflection) with **Silhouette Analysis** (peak score) to choose the optimal cluster count.
4. **Inspect Centroids**: Inverse-transform fitted centroids back to original feature units (`scaler.inverse_transform(kmeans.cluster_centers_)`) for business domain profiling.

---

###  Essential Scikit-Learn Snippet

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit K-Means
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# 3. Evaluate Diagnostics
inertia = kmeans.inertia_
sil_score = silhouette_score(X_scaled, cluster_labels)

print(f'Inertia: {inertia:.2f} | Silhouette Score: {sil_score:.4f}')