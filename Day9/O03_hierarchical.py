'''
Ward's Method (ward): Minimizes the total within-cluster variance. 
                      It tends to form compact, equally-sized spherical clusters and 
                      frequently yields a strong cophenetic score for structured data.

Complete Linkage (complete): Measures distance based on the furthest points between two clusters. 
                             It is sensitive to outliers, which can sometimes distort the 
                             tree structure and lower the coefficient.

Average Linkage (average): Uses the average distance between all pairs of points in two clusters. 
                           It provides a balanced middle-ground, often yielding 
                           stable cophenetic representations.

Single Linkage (single): Uses the closest points between clusters (the "nearest neighbor" approach). 
                         It is prone to the chaining effect (long, strung-out clusters), 
                         which often results in lower cophenetic fidelity.
'''
import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

# Generate a synthetic dataset of 20 points
np.random.seed(42)
X = np.random.rand(20, 2)

# Compute original pairwise distances once
orig_dists = pdist(X)

# Test different linkage methods
methods = ['ward', 'complete', 'average', 'single']

print("--- Cophenetic Correlation by Linkage Method ---")
for method in methods:
    # 1. Compute hierarchical clustering
    linked = linkage(X, method=method)
    
    # 2. Calculate the cophenetic correlation coefficient
    c_coeff, _ = cophenet(linked, orig_dists)
    
    print(f"Linkage: {method.capitalize():<10} | Cophenetic Score: {c_coeff:.4f}")