"""
Title: Unsupervised Customer Segmentation Pipeline with K-Means (Student Exercise)
Description: A modular Python exercise script for loading customer behavioral data,
             scaling features for distance-based algorithms, training a K-Means clustering model,
             evaluating cluster cohesion, profiling segments, and persisting artifacts using joblib.
"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, StandardScaler from sklearn.preprocessing,
#       KMeans from sklearn.cluster, silhouette_score from sklearn.metrics, and joblib
import numpy as np
from numpy.random.mtrand import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery, Setup & Feature Selection
    - Goal: Load 'customer_segmentation_data.csv' and inspect structure.
    - Drop non-analytical identifiers (e.g., 'customer_id').
    - Select numeric feature subset for clustering (e.g., annual income, spending score).
    """
    print("\n[PHASE 1] Discovery & Feature Selection")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    # TODO: Drop ID column ('customer_id')
    # TODO: Select relevant features for clustering (e.g., 'annual_income_k', 'spending_score_1_100', 'purchase_frequency_monthly')

    df = pd.read_csv("customer_segmentation_data.csv")
    x = df.copy()
    x_df = x.drop(columns=["customer_id", "age"])

    print("Columns : \n", x_df.columns.tolist())
    print("-" * 65)
    print("Type: ", type(x_df))
    print("-" * 65)
    print("Head: \n", x_df.head())
    print("-" * 65)

    return x_df


def phase_2_feature_scaling(X):
    """
    [Phase 2] Unsupervised Feature Scaling
    - Goal: Initialize and fit StandardScaler on clustering features.
    - Unsupervised algorithms like K-Means rely heavily on Euclidean distance, making scaling mandatory.
    """
    print("\n[PHASE 2] Unsupervised Feature Scaling")
    print("-" * 65)
    # TODO: Initialize StandardScaler()
    # TODO: Fit and transform features X into X_scaled

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("-" * 65)
    return X_scaled, scaler


def phase_3_kmeans_model_induction(X_scaled, X):
    """
    [Phase 3] K-Means Model Induction
    - Goal: Instantiate KMeans clustering model (e.g., setting n_clusters=3).
    - Fit model to scaled data and assign cluster labels back to dataset.
    """
    print("\n[PHASE 3] K-Means Model Induction & Labeling")
    print("-" * 65)
    # TODO: Instantiate KMeans(n_clusters=3, random_state=42)
    # TODO: Fit model and generate cluster labels (.fit_predict(X_scaled))

    model = KMeans(n_clusters=3, random_state=42)
    X_trained = model.fit_predict(X_scaled)
    df_clustered = X.copy()
    df_clustered["cluster"] = X_trained
    print("Clustering: \n", df_clustered.head(5))
    return model, X_trained, df_clustered


def phase_4_cluster_evaluation_and_profiling(X_scaled, labels, df_original):
    """
    [Phase 4] Cluster Evaluation & Business Profiling
    - Goal: Compute Silhouette Score to measure cluster separation and cohesion.
    - Group original data by assigned cluster labels to profile customer segments.
    """
    print("\n[PHASE 4] Cluster Evaluation & Profiling")
    print("-" * 65)
    # TODO: Compute silhouette_score(X_scaled, labels)
    # TODO: Append cluster labels to original dataframe and compute group means to profile segments

    sil_score = silhouette_score(X_scaled, labels)
    print("Silhouette :\n", sil_score)
    print("-" * 65)
    df_original["cluster"] = labels
    feature_names = df_original.columns.tolist()
    cluster_profile = df_original.groupby("cluster").mean(numeric_only=True)
    print("Cluster Profiled: \n", cluster_profile)
    print("Feature Names : \n", feature_names)
    print("-" * 65)

    return feature_names


def phase_5_joblib_model_persistence(model, scaler, feature_names):
    """
    [Phase 5] Unsupervised Model Persistence with Joblib
    - Goal: Save the trained KMeans model and fitted scaler to disk using joblib.
    - Test loading artifacts to predict the cluster segment of a new customer.
    """
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    # TODO: Save model, scaler, and feature list using joblib.dump()
    # TODO: Load artifacts using joblib.load() and assign a new customer profile to a cluster

    print("Saving...")
    print("-" * 65)
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(feature_names, "features.joblib")

    print("Loading...")
    print("-" * 65)
    load_model = joblib.load("model.joblib")
    load_scaler = joblib.load("scaler.joblib")
    load_feature = joblib.load("features.joblib")

    df_res = [[50, 50, 5]]
    new_df = pd.DataFrame(df_res, columns=load_feature)

    new_df_scaled = load_scaler.transform(new_df)
    print("Prediction :", load_model.predict(new_df_scaled))
    print("-" * 65)


# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================


def run_clustering_pipeline():
    """
    Facade function coordinating the entire unsupervised segmentation pipeline.
    """
    print("=" * 65)
    print("STARTING UNSUPERVISED CLUSTERING LIFECYCLE (K-Means)")
    print("=" * 65)

    # Executing phases chronologically
    X = phase_1_discovery_and_setup()
    X_scaled, scaler = phase_2_feature_scaling(X)
    model, labels, df_clustered = phase_3_kmeans_model_induction(X_scaled, X)
    phase_4_cluster_evaluation_and_profiling(X_scaled, labels, df_clustered)
    phase_5_joblib_model_persistence(model, scaler, list(X.columns))


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_clustering_pipeline()
