"""
Title: Unsupervised Cricket Player Role Segmentation
Description: A modular Python exercise script for loading cricket statistics,
             scaling features, training a K-Means clustering model to group
             players into functional roles (Batsmen, Bowlers, All-Rounders),
             evaluating clusters, and persisting artifacts using joblib.
"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, StandardScaler from sklearn.preprocessing,
#       KMeans from sklearn.cluster, silhouette_score from sklearn.metrics, and joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery, Setup & Feature Selection
    - Goal: Load 'cricket_player_stats.csv' and inspect structure.
    - Separate player names/identifiers from numeric statistics.
    - Select numeric features required for role clustering.
    """
    print("\n[PHASE 1] Discovery & Feature Selection")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    # TODO: Retain 'player_name' for final profiling
    # TODO: Select numerical features (e.g., 'runs_scored', 'batting_average', 'wickets_taken', 'bowling_economy')

    df = pd.read_csv("cricket_player_stats.csv")
    df_num = df.select_dtypes(include=np.number)

    print("Original df: \n", df)
    print("-" * 65)
    print("Numeric df: \n", df_num)
    print("-" * 65)
    return df_num, df


def phase_2_feature_scaling(X):
    """
    [Phase 2] Unsupervised Feature Scaling
    - Goal: Fit StandardScaler on performance metrics to equalize feature weightings.
    """
    print("\n[PHASE 2] Unsupervised Feature Scaling")
    print("-" * 65)
    # TODO: Initialize StandardScaler()
    # TODO: Fit and transform features X into X_scaled

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def phase_3_kmeans_model_induction(X_scaled, df_original):
    """
    [Phase 3] K-Means Model Induction
    - Goal: Train KMeans model setting n_clusters=3 (Batsmen, Bowlers, All-Rounders).
    - Fit model to scaled data and assign cluster labels back to dataset.
    """
    print("\n[PHASE 3] K-Means Model Induction & Labeling")
    print("-" * 65)
    # TODO: Instantiate KMeans(n_clusters=3, random_state=42)
    # TODO: Fit model and generate cluster labels using .fit_predict(X_scaled)

    model = KMeans(n_clusters=3, random_state=42)
    X_trained = model.fit_predict(X_scaled)
    df_clustered = df_original.copy()
    df_clustered["cluster"] = X_trained
    print("Clustered df: \n", df_clustered.head(5))
    print("-" * 65)

    return model, X_trained, df_clustered


def phase_4_cluster_evaluation_and_profiling(X_scaled, labels, df_clustered):
    """
    [Phase 4] Cluster Evaluation & Role Profiling
    - Goal: Calculate Silhouette Score to measure cluster separation.
    - Calculate average statistics per cluster and translate centroids into cricket roles.
    """
    print("\n[PHASE 4] Cluster Evaluation & Role Profiling")
    print("-" * 65)
    # TODO: Compute silhouette_score(X_scaled, labels)
    # TODO: Group df_clustered by 'cluster_id' and compute mean statistics to interpret roles

    sil_score = silhouette_score(X_scaled, labels)
    cluster_profile = df_clustered.groupby("cluster").mean(numeric_only=True)
    feature_names = cluster_profile.columns.tolist()
    print("Silhouette: ", sil_score)
    print("-" * 65)
    print("Profiled df: \n", cluster_profile)
    print("-" * 65)
    print("Feature Names: \n", feature_names)
    print("-" * 65)

    return feature_names


def phase_5_joblib_model_persistence(model, scaler, feature_names):
    """
    [Phase 5] Unsupervised Model Persistence with Joblib
    - Goal: Save the trained KMeans model and scaler to disk using joblib.
    - Test loading artifacts to assign a new player profile to a role cluster.
    """
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    # TODO: Save model, scaler, and feature list using joblib.dump()
    # TODO: Load artifacts using joblib.load() and predict the cluster for a new player profile

    print("Saving...")
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(feature_names, "feature.joblib")

    print("Loading...")
    load_model = joblib.load("model.joblib")
    load_scaler = joblib.load("scaler.joblib")
    load_feature = joblib.load("feature.joblib")

    df_res = [[150, 9000, 50, 7, 5]]
    new_df = pd.DataFrame(df_res, columns=load_feature)

    new_df_scaler = load_scaler.fit_transform(new_df)
    print("Prediction: ", load_model.predict(new_df_scaler))


# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================


def run_cricket_clustering_pipeline():
    """
    Facade function coordinating the cricket player segmentation pipeline.
    """
    print("=" * 65)
    print("STARTING CRICKET PLAYER ROLE CLUSTERING LIFECYCLE (K-Means)")
    print("=" * 65)

    # Executing phases chronologically
    X, df_original = phase_1_discovery_and_setup()
    X_scaled, scaler = phase_2_feature_scaling(X)
    model, labels, df_clustered = phase_3_kmeans_model_induction(X_scaled, df_original)
    phase_4_cluster_evaluation_and_profiling(X_scaled, labels, df_clustered)
    phase_5_joblib_model_persistence(model, scaler, list(X.columns))


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_cricket_clustering_pipeline()

