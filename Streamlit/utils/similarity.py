"""
Player Similarity Computation and Recommendation Module.

Implements Cosine Similarity on scaled player-level performance vectors:
- Precomputes / computes the (2,702 x 2,702) cosine similarity matrix
- Queries top N most statistically similar players
- Extracts individual player profiles and comparative radar/bar metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Default key metrics used for balanced radar profile comparison
RADAR_METRICS = [
    {"col": "Gls", "name": "Goals"},
    {"col": "Ast", "name": "Assists"},
    {"col": "xG", "name": "Exp Goals (xG)"},
    {"col": "xAG", "name": "Exp Assisted Goals (xAG)"},
    {"col": "PrgC", "name": "Prog Carries"},
    {"col": "PrgP", "name": "Prog Passes"},
    {"col": "Sh", "name": "Shots"},
    {"col": "Tkl", "name": "Tackles"},
    {"col": "Int", "name": "Interceptions"},
    {"col": "Touches", "name": "Touches"},
]


def compute_similarity_matrix(X_scaled) -> np.ndarray:
    """
    Computes the pair-wise cosine similarity matrix across all players.
    
    Args:
        X_scaled (pd.DataFrame or np.ndarray): Standardized feature matrix (2702 x 39)
        
    Returns:
        np.ndarray: Symmetric matrix of shape (2702, 2702)
    """
    return cosine_similarity(X_scaled)


def find_similar_players(
    player_name: str,
    player_info: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Identifies the top N players most similar to the selected player using Cosine Similarity.
    
    Args:
        player_name (str): Name of the player
        player_info (pd.DataFrame): DataFrame containing player metadata and cluster assignments
        similarity_matrix (np.ndarray): (2702, 2702) Cosine similarity matrix
        top_n (int): Number of similar players to retrieve (default 5)
        
    Returns:
        pd.DataFrame: Top similar players with columns:
                      ['Rank', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Cluster', 'Cosine Similarity', 'Similarity Pct']
    """
    player_indices = player_info[player_info["Player"] == player_name].index.tolist()
    if not player_indices:
        return pd.DataFrame()

    player_idx = player_indices[0]
    similarities = similarity_matrix[player_idx]

    result = player_info.copy()
    result["Cosine Similarity"] = similarities
    result["Similarity Pct"] = (result["Cosine Similarity"] * 100).round(1)

    # Exclude the selected player themselves
    result = result[result["Player"] != player_name]

    # Sort descending by similarity score
    result = result.sort_values(by="Cosine Similarity", ascending=False).head(top_n).reset_index(drop=True)
    result["Rank"] = [f"#{i + 1}" for i in range(len(result))]

    # Reorder columns cleanly
    front_cols = ["Rank", "Player", "Squad", "Pos", "Age", "Nation", "Cosine Similarity", "Similarity Pct"]
    if "Cluster" in result.columns:
        front_cols.insert(4, "Cluster")

    display_cols = [c for c in front_cols if c in result.columns] + [
        c for c in result.columns if c not in front_cols
    ]
    return result[display_cols]


def get_player_profile(
    player_name: str,
    player_info: pd.DataFrame,
    player_features: pd.DataFrame
) -> dict:
    """
    Retrieves full profile for a given player, combining metadata and raw performance stats.
    """
    info_row = player_info[player_info["Player"] == player_name]
    feat_row = player_features[player_features["Player"] == player_name]

    if info_row.empty:
        return {}

    info = info_row.iloc[0].to_dict()
    stats = feat_row.iloc[0].to_dict() if not feat_row.empty else {}

    return {
        "metadata": info,
        "statistics": stats,
    }


def prepare_radar_comparison_data(
    selected_player: str,
    similar_players: list,
    player_features: pd.DataFrame,
    metrics_list: list = None
) -> pd.DataFrame:
    """
    Prepares min-max normalized (0-100 scale) statistics across selected metrics
    so that radar/spider charts can visually compare different player roles without scale distortion.
    
    Args:
        selected_player (str): Name of base player
        similar_players (list): Names of compared players (e.g. Top 1 or Top 5)
        player_features (pd.DataFrame): Aggregated stats dataframe
        metrics_list (list): List of metric column names (defaults to RADAR_METRICS)
        
    Returns:
        pd.DataFrame: Normalized metrics for each player with values in range [0, 100]
    """
    if metrics_list is None:
        metrics_list = [m["col"] for m in RADAR_METRICS if m["col"] in player_features.columns]

    target_players = [selected_player] + [p for p in similar_players if p != selected_player]
    subset = player_features[player_features["Player"].isin(target_players)].copy()

    # Calculate min and max from entire population for fair percentile/scaled representation
    min_vals = player_features[metrics_list].min()
    max_vals = player_features[metrics_list].max()

    # Avoid zero division
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0

    normalized_df = pd.DataFrame()
    normalized_df["Player"] = subset["Player"].values

    for col in metrics_list:
        normalized_df[col] = (
            ((subset[col].values - min_vals[col]) / range_vals[col]) * 100
        ).round(1)

    return normalized_df
