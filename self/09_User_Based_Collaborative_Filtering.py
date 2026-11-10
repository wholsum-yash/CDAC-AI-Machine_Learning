# ---------- Q9: User-based collaborative filtering ----------
# NOTE: ratings below are SYNTHETIC (illustrative), not real user data.
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# 1. Small ratings table with missing (unrated) entries
ratings = pd.DataFrame({
    "user_id":  [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5],
    "movie_id": ["M1","M2","M3","M1","M3","M4","M5","M2","M3","M5",
                 "M1","M2","M4","M1","M3","M4","M5"],
    "rating":   [5, 4, 3, 4, 5, 2, 3, 3, 4, 5, 5, 4, 3, 4, 5, 4, 2],
})

# 2. Split observed ratings into train / test (so test never leaks)
train_df, test_df = train_test_split(ratings, test_size=0.25, random_state=42)

# 3. User-movie matrix from TRAINING ratings only
matrix = train_df.pivot(index="user_id", columns="movie_id", values="rating")
print("Training user-movie matrix:\n", matrix)

# 4. User-user similarity (fill unseen with 0, then cosine similarity)
sim = cosine_similarity(matrix.fillna(0))
sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
print("\nUser similarity:\n", sim_df.round(2))

def predict_rating(user, movie):
    """Weighted average of other users' ratings, weighted by similarity."""
    if movie not in matrix.columns:
        return np.nan
    raters = matrix[movie].notna()          # users who rated this movie
    weights = sim_df.loc[user, raters]
    if weights.sum() == 0:
        return np.nan
    return float(np.dot(weights, matrix.loc[raters, movie]) / weights.sum())

# 5. Top-3 recommendations for each user
recommendations = {}
for user in matrix.index:
    preds = {m: predict_rating(user, m)
             for m in matrix.columns if pd.isna(matrix.loc[user, m])}
    preds = {m: p for m, p in preds.items() if not np.isnan(p)}
    recommendations[user] = sorted(preds.items(), key=lambda x: -x[1])[:3]
    print(f"User {user} -> top 3: {recommendations[user]}")

# 6. Evaluate with Precision@K and Recall@K against held-out ratings
K = 3
RELEVANT = 4.0   # a rating >= 4 counts as "relevant"

precisions, recalls = [], []
for user in matrix.index:
    actual = test_df[(test_df["user_id"] == user) &
                     (test_df["rating"] >= RELEVANT)]["movie_id"].tolist()
    if not actual:
        continue
    recs = [m for m, _ in recommendations[user][:K]]
    hits = len(set(recs) & set(actual))
    precisions.append(hits / K)
    recalls.append(hits / len(actual))

print(f"\nPrecision@{K}: {np.mean(precisions):.3f}")
print(f"Recall@{K}:    {np.mean(recalls):.3f}")
