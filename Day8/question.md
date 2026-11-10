### Question 1
Why is feature scaling (e.g., `StandardScaler`) essential before applying K-Means clustering?
- [ ] A) It speeds up file I/O operations.
- [ ] B) K-Means relies on Euclidean distance, so unscaled features with larger magnitude ranges will disproportionately dominate cluster assignments.
- [ ] C) K-Means cannot compute categorical variables without scaling.
- [ ] D) Scaling guarantees that the algorithm will always find the global minimum.

**Correct Answer**: **B**  
*Explanation*: K-Means calculates distance vectors using $L_2$ norm. If one variable ranges from $1$ to $100,000$ and another from $0.1$ to $1.0$, the former will dictate all distance measurements unless scaled.

---

### Question 2
What problem does the **K-Means++** algorithm specifically address?
- [ ] A) Finding the optimal value of $K$ automatically.
- [ ] B) Eliminating the need for feature scaling.
- [ ] C) Choosing initial centroid locations strategically to avoid bad local minima and speed up convergence.
- [ ] D) Handling non-spherical spatial density clusters.

**Correct Answer**: **C**  
*Explanation*: K-Means++ picks initial centroids probabilistically based on squared distance from existing centroids, preventing cluster collapse into poor local optima.
---

### Question 3
What does a Silhouette Score of $+0.85$ indicate for a clustered dataset?
- [ ] A) Points are incorrectly assigned to neighboring clusters.
- [ ] B) Clusters are dense, well-separated, and clearly distinct from one another.
- [ ] C) The optimal number of clusters is $K=1$.
- [ ] D) The algorithm has failed to converge within `max_iter`.

**Correct Answer**: **B**  
*Explanation*: The Silhouette score ranges from $-1$ to $+1$. Values near $+1$ represent high within-cluster cohesion and strong between-cluster separation.
---

### Question 4
Which of the following scenarios is K-Means NOT suitable for?
- [ ] A) Grouping customer spending behavior into 4 spherical clusters.
- [ ] B) Segmenting tabular data after applying `StandardScaler`.
- [ ] C) Clustering concentric circle datasets or arbitrary non-convex density shapes.
- [ ] D) Quantizing pixel colors in image compression.

**Correct Answer**: **C**  
*Explanation*: K-Means uses Voronoi partitioning, forcing linear decision boundaries between centroids. It cannot separate non-linearly separable or manifold structures (like concentric rings).