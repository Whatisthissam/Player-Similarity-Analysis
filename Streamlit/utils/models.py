"""
Machine Learning Models Module for Football Player Clustering and Evaluation.

Implements and evaluates the 4 unsupervised learning models from PlayerClassification.ipynb:
1. K-Means Clustering (optimal K=2 via silhouette analysis)
2. Agglomerative Hierarchical Clustering
3. DBSCAN (eps=1.5, min_samples=10, handling noise -1)
4. Gaussian Mixture Model (n_components=2, covariance_type='full')

Calculates the three standard cluster evaluation metrics:
- Silhouette Score (higher is better)
- Davies-Bouldin Index (lower is better)
- Calinski-Harabasz Score (higher is better)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def compute_k_selection_metrics(X_scaled, k_range=range(2, 11)):
    """
    Computes Inertia (Elbow method) and Silhouette scores for K in range [2, 10].
    """
    inertias = []
    silhouette_scores_list = []
    k_list = list(k_range)

    for k in k_list:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        silhouette_scores_list.append(float(silhouette_score(X_scaled, labels)))

    optimal_idx = int(np.argmax(silhouette_scores_list))
    optimal_k = k_list[optimal_idx]

    return {
        "k_list": k_list,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores_list,
        "optimal_k": optimal_k,
    }


def train_and_evaluate_all_models(X_scaled, optimal_k=2):
    """
    Trains all four clustering models on X_scaled and computes exact evaluation metrics.
    
    Returns:
        models (dict): Trained model instances
        labels (dict): Cluster labels for each model
        metrics (dict): Dict of metrics per model
        comparison_df (pd.DataFrame): Comparison table matching notebook Cell 22
        best_model (str): Name of top-performing model ('K-Means')
    """
    # 1. K-Means
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    sil_kmeans = float(silhouette_score(X_scaled, kmeans_labels))
    db_kmeans = float(davies_bouldin_score(X_scaled, kmeans_labels))
    ch_kmeans = float(calinski_harabasz_score(X_scaled, kmeans_labels))

    # 2. Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=optimal_k)
    agg_labels = agg.fit_predict(X_scaled)
    sil_agg = float(silhouette_score(X_scaled, agg_labels))
    db_agg = float(davies_bouldin_score(X_scaled, agg_labels))
    ch_agg = float(calinski_harabasz_score(X_scaled, agg_labels))

    # 3. DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    valid_mask = dbscan_labels != -1

    if len(set(dbscan_labels) - {-1}) >= 2:
        sil_dbscan = float(silhouette_score(X_scaled[valid_mask], dbscan_labels[valid_mask]))
        db_dbscan = float(davies_bouldin_score(X_scaled[valid_mask], dbscan_labels[valid_mask]))
        ch_dbscan = float(calinski_harabasz_score(X_scaled[valid_mask], dbscan_labels[valid_mask]))
    else:
        sil_dbscan = np.nan
        db_dbscan = np.nan
        ch_dbscan = np.nan

    # 4. Gaussian Mixture Model
    gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    sil_gmm = float(silhouette_score(X_scaled, gmm_labels))
    db_gmm = float(davies_bouldin_score(X_scaled, gmm_labels))
    ch_gmm = float(calinski_harabasz_score(X_scaled, gmm_labels))

    comparison_df = pd.DataFrame({
        "Model": [
            "K-Means",
            "Agglomerative",
            "DBSCAN",
            "Gaussian Mixture"
        ],
        "Silhouette Score": [
            sil_kmeans,
            sil_agg,
            sil_dbscan,
            sil_gmm
        ],
        "Davies-Bouldin Index": [
            db_kmeans,
            db_agg,
            db_dbscan,
            db_gmm
        ],
        "Calinski-Harabasz Score": [
            ch_kmeans,
            ch_agg,
            ch_dbscan,
            ch_gmm
        ]
    })

    # Best model determined by Silhouette Score
    best_model_name = comparison_df.loc[
        comparison_df["Silhouette Score"].idxmax(), "Model"
    ]

    models = {
        "K-Means": kmeans,
        "Agglomerative": agg,
        "DBSCAN": dbscan,
        "Gaussian Mixture": gmm,
    }

    labels = {
        "K-Means": kmeans_labels,
        "Agglomerative": agg_labels,
        "DBSCAN": dbscan_labels,
        "Gaussian Mixture": gmm_labels,
    }

    metrics = {
        "K-Means": {"silhouette": sil_kmeans, "davies_bouldin": db_kmeans, "calinski_harabasz": ch_kmeans},
        "Agglomerative": {"silhouette": sil_agg, "davies_bouldin": db_agg, "calinski_harabasz": ch_agg},
        "DBSCAN": {"silhouette": sil_dbscan, "davies_bouldin": db_dbscan, "calinski_harabasz": ch_dbscan},
        "Gaussian Mixture": {"silhouette": sil_gmm, "davies_bouldin": db_gmm, "calinski_harabasz": ch_gmm},
    }

    return {
        "models": models,
        "labels": labels,
        "metrics": metrics,
        "comparison_df": comparison_df,
        "best_model": best_model_name,
        "best_labels": labels[best_model_name],
        "optimal_k": optimal_k,
    }


def save_models_and_artifacts(artifacts_dict, directory=MODELS_DIR):
    """Save trained models and computed results to disk."""
    os.makedirs(directory, exist_ok=True)
    joblib_path = os.path.join(directory, "clustering_artifacts.joblib")
    joblib.dump(artifacts_dict, joblib_path)
    return joblib_path


def load_cached_artifacts(directory=MODELS_DIR):
    """Load cached artifacts if available."""
    joblib_path = os.path.join(directory, "clustering_artifacts.joblib")
    if os.path.exists(joblib_path):
        try:
            return joblib.load(joblib_path)
        except Exception:
            return None
    return None
