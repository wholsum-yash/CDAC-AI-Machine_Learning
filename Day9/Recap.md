### K-Means Diagnostic 
    Inertia (Elbow Method) 
    Silhouette Score 

    * **Inertia**:
  $$\text{Inertia} = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2$$
* **Silhouette Coefficient** (for point $i$):
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  Where $a(i)$ is mean intra-cluster distance, and $b(i)$ is mean nearest-cluster distance.