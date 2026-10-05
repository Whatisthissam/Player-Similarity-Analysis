"""
Data Preprocessing and Pipeline Utilities for Football Player Similarity Analysis.

Replicates the exact preprocessing workflow from PlayerClassification.ipynb:
1. Load raw dataset (2,854 records)
2. Data cleaning (remove duplicate rows, empty columns, infinite values)
3. Column filtering (remove Rk and repeated metadata like Pos_, Squad_, etc.)
4. Feature selection (39 key performance features)
5. Player-level aggregation (group by Player and calculate mean performance)
6. Alignment of player metadata (Nation, Pos, Squad, Age)
7. Missing value imputation using feature medians (yielding 2,702 unique players)
8. Feature scaling using StandardScaler
9. PCA dimensionality reduction (PC1 ~49.36%, PC2 ~19.01%, Total ~68.38%)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# The exact 39 performance features selected in the notebook
PERFORMANCE_FEATURES = [
    "Age", "MP", "Starts", "Min", "90s", "Gls", "Ast", "G+A", "G-PK",
    "xG", "xAG", "npxG+xAG", "PrgC", "PrgP", "PrgR", "Sh", "SoT", "SoT%",
    "Cmp", "Att", "Cmp%", "xA", "KP", "PPA", "SCA", "SCA90", "Tkl", "TklW",
    "Int", "Tkl+Int", "Clr", "Touches", "Succ", "Succ%", "Carries", "CPA",
    "Rec", "Won", "Won%"
]

# Features selected for correlation heatmap exploration
CORRELATION_FEATURES = [
    "Age", "MP", "Starts", "Min",
    "Gls", "Ast", "xG", "xAG",
    "PrgC", "PrgP", "PrgR",
    "Sh", "SoT", "Cmp", "Att",
    "KP", "PPA", "SCA",
    "Tkl", "Int", "Clr",
    "Touches", "Carries"
]

METADATA_KEYWORDS = [
    "Nation_", "Pos_", "Squad_", "Comp_",
    "Age_", "Born_", "Player_"
]


def resolve_data_path(custom_path: str = None) -> str:
    """Find the path to players_data-2024_2025.csv robustly."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "players_data-2024_2025.csv"),
        os.path.join("Streamlit", "data", "players_data-2024_2025.csv"),
        os.path.join("data", "players_data-2024_2025.csv"),
        os.path.join("Dataset", "players_data-2024_2025.csv"),
        os.path.join("..", "Dataset", "players_data-2024_2025.csv"),
        os.path.join(os.path.dirname(__file__), "..", "..", "Dataset", "players_data-2024_2025.csv"),
    ]

    for p in candidate_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            return abs_p

    raise FileNotFoundError(
        "Could not locate 'players_data-2024_2025.csv'. "
        "Please ensure the dataset exists in Streamlit/data/ or Dataset/."
    )


def load_raw_data(data_path: str = None) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    filepath = resolve_data_path(data_path)
    df = pd.read_csv(filepath)
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataset:
    - Drops exact duplicate rows
    - Drops completely empty columns
    - Replaces infinite values with NaN
    - Drops 'Rk' and repeated metadata columns (e.g. Squad_*, Pos_*)
    """
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(axis=1, how="all")
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    ranking_columns = [
        col for col in df_clean.columns
        if col == "Rk" or col.startswith("Rk_")
    ]

    repeated_metadata = [
        col for col in df_clean.columns
        if any(col.startswith(keyword) for keyword in METADATA_KEYWORDS)
    ]

    columns_to_remove = list(set(ranking_columns + repeated_metadata))
    df_clean = df_clean.drop(columns=columns_to_remove, errors="ignore")
    return df_clean


def prepare_player_data(df_clean: pd.DataFrame):
    """
    Prepares aggregated player-level statistics and metadata:
    - Filters to existing performance features (39)
    - Converts performance features to numeric
    - Aggregates multiple records by Player using mean()
    - Aligns player_info (Player, Nation, Pos, Squad, Age)
    - Imputes missing values in feature matrix X using medians
    
    Returns:
        player_info (pd.DataFrame): 2,702 unique players with metadata
        player_features (pd.DataFrame): 2,702 unique players with raw stats
        X (pd.DataFrame): 2,702 x 39 imputed feature matrix
        valid_features (list): list of 39 feature column names
    """
    valid_features = [col for col in PERFORMANCE_FEATURES if col in df_clean.columns]

    player_features = df_clean[["Player"] + valid_features].copy()
    player_features[valid_features] = player_features[valid_features].apply(
        pd.to_numeric, errors="coerce"
    )
    player_features = player_features.replace([np.inf, -np.inf], np.nan)

    # Aggregate multiple records by Player using mean
    player_features = (
        player_features
        .groupby("Player", as_index=False)[valid_features]
        .mean()
    )

    # Extract single metadata record per player
    metadata_cols = ["Player", "Nation", "Pos", "Squad", "Age"]
    avail_meta = [c for c in metadata_cols if c in df_clean.columns]
    player_info = (
        df_clean[avail_meta]
        .drop_duplicates(subset="Player")
        .reset_index(drop=True)
    )

    # Match player order
    player_info = player_info[
        player_info["Player"].isin(player_features["Player"])
    ].copy()
    player_info = (
        player_info
        .set_index("Player")
        .loc[player_features["Player"]]
        .reset_index()
    )

    # Feature matrix X
    X = player_features[valid_features].copy()
    X = X.fillna(X.median())
    X = X.reset_index(drop=True)
    player_info = player_info.reset_index(drop=True)

    return player_info, player_features, X, valid_features


def scale_features(X: pd.DataFrame, feature_names: list):
    """Fit StandardScaler and transform feature matrix."""
    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(
        X_scaled_arr,
        columns=feature_names,
        index=X.index
    )
    return scaler, X_scaled_df


def compute_pca(X_scaled_df: pd.DataFrame, n_components: int = 2):
    """
    Compute 2-component PCA for dimensionality reduction and visualization.
    Returns:
        pca (PCA): Fitted PCA model
        pca_df (pd.DataFrame): DataFrame with columns ['PC1', 'PC2']
        explained_variance (dict): Breakdown of variance explained
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled_df)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=X_scaled_df.index)

    evr = pca.explained_variance_ratio_
    explained_variance = {
        "PC1": float(evr[0]),
        "PC2": float(evr[1]),
        "Total": float(evr.sum()),
        "PC1_pct": f"{evr[0] * 100:.2f}%",
        "PC2_pct": f"{evr[1] * 100:.2f}%",
        "Total_pct": f"{evr.sum() * 100:.2f}%",
    }

    return pca, pca_df, explained_variance


def get_full_pipeline_data(data_path: str = None):
    """
    Runs the complete preprocessing and PCA pipeline.
    Returns a unified dictionary containing all dataframes, models, and metadata.
    """
    raw_df = load_raw_data(data_path)
    df_clean = clean_raw_data(raw_df)
    player_info, player_features, X, valid_features = prepare_player_data(df_clean)
    scaler, X_scaled = scale_features(X, valid_features)
    pca, pca_df, explained_variance = compute_pca(X_scaled, n_components=2)

    return {
        "raw_df": raw_df,
        "df_clean": df_clean,
        "player_info": player_info,
        "player_features": player_features,
        "X": X,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "pca": pca,
        "pca_df": pca_df,
        "explained_variance": explained_variance,
        "performance_features": valid_features,
        "raw_rows": len(raw_df),
        "unique_players": len(player_info),
        "feature_count": len(valid_features),
    }
