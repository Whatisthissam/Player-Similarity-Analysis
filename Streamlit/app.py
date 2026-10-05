"""
Football Player Similarity Analysis - Streamlit Web Application
Case Study 108: Machine Learning Major Project

An end-to-end unsupervised machine learning and player recommendation platform.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Add current directory to path for clean modular imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from utils.preprocessing import get_full_pipeline_data, resolve_data_path
from utils.models import (
    train_and_evaluate_all_models,
    compute_k_selection_metrics,
    save_models_and_artifacts,
    load_cached_artifacts
)
from utils.similarity import (
    compute_similarity_matrix,
    find_similar_players,
    get_player_profile,
    prepare_radar_comparison_data,
    RADAR_METRICS
)
from utils.visualization import (
    THEME,
    plot_position_distribution,
    plot_metric_distribution,
    plot_top_players_metric,
    plot_goals_vs_assists,
    plot_correlation_heatmap,
    plot_elbow_and_silhouette,
    plot_model_metric_bar,
    plot_pca_clusters,
    plot_player_radar,
    plot_metric_comparison_bars
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Player Similarity Analysis | ML Major Project",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_custom_css():
    css_path = os.path.join(CURRENT_DIR, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback inline basic style if file not reached
        st.markdown(
            """
            <style>
            .kpi-card { background: #1e293b; padding: 1.2rem; border-radius: 10px; border-left: 4px solid #10b981; }
            .kpi-val { font-size: 2rem; font-weight: 700; color: #fff; }
            </style>
            """,
            unsafe_allow_html=True
        )

load_custom_css()

# -----------------------------------------------------------------------------
# CACHED DATA PIPELINE & MODELS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_app_data():
    try:
        return get_full_pipeline_data()
    except Exception as e:
        st.error(f"Error loading and preprocessing dataset: {str(e)}")
        st.stop()

@st.cache_resource(show_spinner=False)
def load_app_models(X_scaled_df):
    try:
        # Check if saved artifacts exist on disk
        cached = load_cached_artifacts()
        if cached is not None:
            return cached
        # Otherwise train dynamically and save
        results = train_and_evaluate_all_models(X_scaled_df, optimal_k=2)
        save_models_and_artifacts(results)
        return results
    except Exception as e:
        st.error(f"Error training clustering models: {str(e)}")
        st.stop()

@st.cache_resource(show_spinner=False)
def get_similarity_matrix(X_scaled_df):
    return compute_similarity_matrix(X_scaled_df)

@st.cache_resource(show_spinner=False)
def get_k_analysis(X_scaled_df):
    return compute_k_selection_metrics(X_scaled_df)

# Initialize pipeline
with st.spinner("Initializing Player Similarity Engine..."):
    data = load_app_data()
    models_res = load_app_models(data["X_scaled"])
    sim_matrix = get_similarity_matrix(data["X_scaled"])

# Attach best model cluster assignments to player_info
data["player_info"]["Cluster"] = models_res["best_labels"]

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.2rem;">
        <span style="font-size: 2rem;">⚽</span>
        <div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">SCOUT-AI</div>
            <div style="font-size: 0.75rem; color: #10b981; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Player Similarity Analytics</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

pages = [
    "🏠 Dashboard",
    "👤 Player Similarity",
    "📊 Player Analytics",
    "🧠 ML Models",
    "📈 Model Comparison",
    "🔬 PCA Visualization",
    "📋 Dataset Explorer",
    "ℹ️ About Project"
]

selected_page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="background: rgba(30, 41, 59, 0.6); padding: 0.85rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); font-size: 0.8rem;">
        <div style="color: #94a3b8; font-weight: 600; margin-bottom: 0.3rem;">SYSTEM STATUS</div>
        <div style="color: #e2e8f0;">● Records: <b>{data['raw_rows']:,}</b></div>
        <div style="color: #e2e8f0;">● Players: <b>{data['unique_players']:,}</b></div>
        <div style="color: #e2e8f0;">● Features: <b>{data['feature_count']}</b></div>
        <div style="color: #10b981; margin-top: 0.4rem; font-weight: 600;">✓ Model: K-Means (K=2)</div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# PAGE 1: 🏠 DASHBOARD
# -----------------------------------------------------------------------------
if selected_page == "🏠 Dashboard":
    st.markdown('<div class="main-header">PLAYER SIMILARITY ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Machine Learning Based Football Player Recommendation System</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-callout">
            <b>Project Summary:</b> This sports analytics platform evaluates professional football player performance using unsupervised machine learning 
            and high-dimensional clustering. By computing pairwise <b>Cosine Similarity</b> across 39 standardized statistical features, 
            it uncovers natural player archetypes and provides data-driven scouting recommendations and player replacement alternatives.
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI Metric Cards
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Unique Players</div>
                <div class="kpi-value">{data['unique_players']:,}</div>
                <span class="kpi-badge">Aggregated from {data['raw_rows']:,} rows</span>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Performance Features</div>
                <div class="kpi-value">{data['feature_count']}</div>
                <span class="kpi-badge">Attacking, Passing, Defending</span>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">ML Models Evaluated</div>
                <div class="kpi-value">4</div>
                <span class="kpi-badge">Unsupervised Clustering</span>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Similarity Method</div>
                <div class="kpi-value" style="font-size: 1.45rem; padding-top: 0.2rem;">Cosine Similarity</div>
                <span class="kpi-badge">Matrix: 2,702 × 2,702</span>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Optimal Model</div>
                <div class="kpi-value" style="color: #00e676;">K-Means</div>
                <span class="kpi-badge">Silhouette: 0.3592 (K=2)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Workflow Visual Pipeline
    st.markdown("### 🔄 Machine Learning Workflow Architecture")
    st.markdown(
        """
        <div class="workflow-container">
            <div class="workflow-step active">📁 Raw Dataset (2,854)</div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-step active">🧹 Data Cleaning & Aggregation</div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-step active">⚖️ StandardScaler (39 feats)</div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-step active">🔬 PCA (68.38% Var)</div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-step active">🧠 4 Cluster Models</div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-step active">📐 Cosine Similarity</div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-step active" style="border-color: #00e676; background: rgba(0,230,118,0.2);">🎯 Top 5 Similar Players</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dashboard Quick Insights Row
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### ⚽ Playing Positions Distribution")
        fig_pos = plot_position_distribution(data["player_info"])
        st.plotly_chart(fig_pos, use_container_width=True)

    with col2:
        st.markdown("#### 🔬 PCA 2D Cluster Space (K-Means)")
        fig_pca_mini = plot_pca_clusters(
            data["pca_df"],
            data["player_info"],
            models_res["best_labels"],
            model_name="K-Means"
        )
        st.plotly_chart(fig_pca_mini, use_container_width=True)

    # Featured Star Players Showcase
    st.markdown("---")
    st.markdown("### 🌟 Featured Players Spotlight")
    st.caption("Click to jump straight to similarity analysis for elite profiles:")

    star_cols = st.columns(4)
    featured_stars = [
        {"name": "Erling Haaland", "squad": "Manchester City", "pos": "FW", "role": "Pure Goalscorer"},
        {"name": "Kylian Mbappé", "squad": "Real Madrid", "pos": "FW", "role": "Dynamic Forward"},
        {"name": "Mohamed Salah", "squad": "Liverpool", "pos": "FW", "role": "Attacking Winger"},
        {"name": "Bukayo Saka", "squad": "Arsenal", "pos": "FW", "role": "Creative Wide Forward"}
    ]

    for i, star in enumerate(featured_stars):
        with star_cols[i]:
            st.markdown(
                f"""
                <div style="background: #111827; border: 1px solid #374151; border-radius: 10px; padding: 1rem; text-align: center;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">{star['name']}</div>
                    <div style="font-size: 0.82rem; color: #38bdf8; font-weight: 600;">{star['squad']}</div>
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.3rem;">Pos: {star['pos']} • {star['role']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# -----------------------------------------------------------------------------
# PAGE 2: 👤 PLAYER SIMILARITY (CORE RECOMMENDATION ENGINE)
# -----------------------------------------------------------------------------
elif selected_page == "👤 Player Similarity":
    st.markdown('<div class="main-header">PLAYER SIMILARITY RECOMMENDATION ENGINE</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Select any player to inspect statistical profile and discover Top 5 statistically similar players using Cosine Similarity</div>', unsafe_allow_html=True)

    all_players = sorted(data["player_info"]["Player"].unique().tolist())

    # Filter / Search controls
    search_col1, search_col2, search_col3 = st.columns([2, 1, 1])
    with search_col1:
        # Default to Erling Haaland if available
        default_index = all_players.index("Erling Haaland") if "Erling Haaland" in all_players else 0
        selected_player = st.selectbox(
            "Select a Player (Type to search)",
            options=all_players,
            index=default_index,
            key="player_select_box"
        )
    with search_col2:
        top_n_select = st.selectbox("Recommendations Count", [5, 10], index=0)
    with search_col3:
        position_filter = st.selectbox(
            "Filter Recommendations by Position",
            ["All Positions", "FW", "MF", "DF", "GK"],
            index=0
        )

    if not selected_player:
        st.warning("Please select a player to continue.")
        st.stop()

    # Retrieve Player Profile & Top 5
    profile = get_player_profile(selected_player, data["player_info"], data["player_features"])
    meta = profile.get("metadata", {})
    stats = profile.get("statistics", {})

    cluster_id = meta.get("Cluster", 0)

    # Top Banner Player Card
    st.markdown(
        f"""
        <div class="player-banner">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                <div>
                    <h2 class="player-name">{selected_player}</h2>
                    <div class="player-meta-row">
                        <span class="meta-chip club">🏟️ {meta.get('Squad', 'Unknown Club')}</span>
                        <span class="meta-chip pos">⚽ Position: {meta.get('Pos', 'N/A')}</span>
                        <span class="meta-chip">🌍 {meta.get('Nation', 'N/A')}</span>
                        <span class="meta-chip">🎂 Age: {meta.get('Age', 'N/A')}</span>
                        <span class="meta-chip cluster">🏷️ K-Means Cluster {cluster_id}</span>
                    </div>
                </div>
                <div style="text-align: right; margin-top: 0.5rem;">
                    <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Vector Dimension</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #38bdf8;">39 Features</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Key Performance Metrics Ribbon
    stat_cols = st.columns(6)
    key_metrics_display = [
        ("Minutes", stats.get("Min", 0), "{:.0f}"),
        ("Goals", stats.get("Gls", 0), "{:.0f}"),
        ("Assists", stats.get("Ast", 0), "{:.0f}"),
        ("Expected Goals (xG)", stats.get("xG", 0), "{:.2f}"),
        ("Prog Passes (PrgP)", stats.get("PrgP", 0), "{:.1f}"),
        ("Tackles (Tkl)", stats.get("Tkl", 0), "{:.1f}"),
    ]
    for i, (label, val, fmt) in enumerate(key_metrics_display):
        with stat_cols[i]:
            val_str = fmt.format(val) if pd.notnull(val) else "0"
            st.metric(label=label, value=val_str)

    # Compute Recommendations
    similar_players_df = find_similar_players(
        selected_player,
        data["player_info"],
        sim_matrix,
        top_n=20 if position_filter != "All Positions" else top_n_select
    )

    if position_filter != "All Positions":
        similar_players_df = similar_players_df[similar_players_df["Pos"].str.contains(position_filter, na=False)]
        similar_players_df = similar_players_df.head(top_n_select).reset_index(drop=True)
        similar_players_df["Rank"] = [f"#{i+1}" for i in range(len(similar_players_df))]

    st.markdown("---")
    st.markdown(f"### 🎯 Top {len(similar_players_df)} Statistically Similar Players")
    st.caption("Calculated using Cosine Similarity on Standardized 39-dimensional Performance Vectors:")

    # Top Recommendations Layout: Visual Cards + Table
    rec_col, chart_col = st.columns([1.1, 1.3])

    with rec_col:
        for _, row in similar_players_df.iterrows():
            rank_num = row['Rank'].replace('#', '')
            rank_class = f"rank-{rank_num}" if int(rank_num) <= 5 else "rank-5"
            st.markdown(
                f"""
                <div class="similar-card">
                    <div class="similar-left">
                        <div class="rank-pill {rank_class}">{row['Rank']}</div>
                        <div class="similar-info">
                            <h4>{row['Player']}</h4>
                            <p><b>{row['Squad']}</b> • {row['Pos']} • Age: {row['Age']} • Cluster: {row.get('Cluster', 'N/A')}</p>
                        </div>
                    </div>
                    <div class="sim-score-container">
                        <div class="sim-score-pct">{row['Similarity Pct']:.1f}%</div>
                        <div class="sim-score-label">Cosine Similarity</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with chart_col:
        # Radar Comparison Chart
        if not similar_players_df.empty:
            top_similar_name = similar_players_df.iloc[0]["Player"]
            radar_data = prepare_radar_comparison_data(
                selected_player,
                similar_players_df["Player"].tolist()[:3],
                data["player_features"]
            )
            fig_radar = plot_player_radar(radar_data, selected_player, top_similar_name)
            st.plotly_chart(fig_radar, use_container_width=True)

    # Detailed Grouped Bar Chart Comparison
    st.markdown("---")
    st.markdown("### 📊 Head-to-Head Feature Comparison")
    metric_options = [
        "Gls", "Ast", "xG", "xAG", "PrgC", "PrgP", "PrgR", "Sh", "SoT", "Cmp%", "Tkl", "Int", "Clr", "Touches"
    ]
    chosen_metrics = st.multiselect(
        "Select Metrics for Head-to-Head Comparison:",
        options=metric_options,
        default=["Gls", "Ast", "xG", "xAG", "PrgC", "PrgP", "Tkl", "Int"]
    )
    if chosen_metrics and not similar_players_df.empty:
        fig_bars = plot_metric_comparison_bars(
            data["player_features"],
            selected_player,
            similar_players_df["Player"].tolist(),
            selected_metrics=chosen_metrics
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    # Full stats comparison table
    with st.expander("📋 View Side-by-Side Raw Statistics Table"):
        compared_names = [selected_player] + similar_players_df["Player"].tolist()
        comp_subset = data["player_features"][data["player_features"]["Player"].isin(compared_names)].copy()
        # Merge metadata without duplicating Age
        info_cols = [c for c in ["Player", "Squad", "Pos", "Nation", "Cluster"] if c in data["player_info"].columns]
        comp_subset = comp_subset.merge(data["player_info"][info_cols], on="Player")
        front = [c for c in ["Player", "Squad", "Pos", "Age", "Cluster"] if c in comp_subset.columns]
        reordered = front + [c for c in comp_subset.columns if c not in front]
        st.dataframe(comp_subset[reordered].set_index("Player"), use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 3: 📊 PLAYER ANALYTICS
# -----------------------------------------------------------------------------
elif selected_page == "📊 Player Analytics":
    st.markdown('<div class="main-header">EXPLORATORY PLAYER ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Statistical distribution and correlation analysis of the 2,702 unique players</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🏆 Leaderboards", "🔥 Feature Correlations"])

    with tab1:
        st.markdown("#### Univariate Statistical Distributions")
        c1, c2 = st.columns(2)
        with c1:
            fig_goals = plot_metric_distribution(data["player_features"], "Gls", "Goals Scored", color=THEME["pitch_green"])
            st.plotly_chart(fig_goals, use_container_width=True)
        with c2:
            fig_ast = plot_metric_distribution(data["player_features"], "Ast", "Assists Provided", color=THEME["cyber_blue"])
            st.plotly_chart(fig_ast, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig_age = plot_metric_distribution(data["player_info"], "Age", "Player Age", color=THEME["purple"])
            st.plotly_chart(fig_age, use_container_width=True)
        with c4:
            fig_min = plot_metric_distribution(data["player_features"], "Min", "Minutes Played", color=THEME["amber"])
            st.plotly_chart(fig_min, use_container_width=True)

    with tab2:
        st.markdown("#### Performance Leaders")
        l1, l2 = st.columns(2)
        with l1:
            fig_top_g = plot_top_players_metric(data["player_features"], "Gls", "Top 15 Goal Scorers", color=THEME["bright_green"])
            st.plotly_chart(fig_top_g, use_container_width=True)
        with l2:
            fig_top_a = plot_top_players_metric(data["player_features"], "Ast", "Top 15 Assist Leaders", color=THEME["cyber_blue"])
            st.plotly_chart(fig_top_a, use_container_width=True)

        st.markdown("#### Goals vs. Assists Bivariate Relationship")
        fig_scatter = plot_goals_vs_assists(data["player_features"].merge(data["player_info"][["Player", "Squad", "Pos"]], on="Player"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.markdown("#### Feature Correlation Matrix")
        st.caption("Pearson correlation between 23 key metrics selected from offensive, creative, and defensive areas:")
        fig_heat = plot_correlation_heatmap(data["player_features"])
        st.plotly_chart(fig_heat, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 4: 🧠 ML MODELS
# -----------------------------------------------------------------------------
elif selected_page == "🧠 ML Models":
    st.markdown('<div class="main-header">UNSUPERVISED MACHINE LEARNING MODELS</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Detailed evaluation of the 4 clustering algorithms trained on the scaled 39-feature dataset</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid #10b981; border-radius: 10px; padding: 1.2rem; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <span style="font-size: 1.5rem;">🏆</span>
                <span style="font-size: 1.25rem; font-weight: 800; color: #00e676;">BEST PERFORMING MODEL: K-MEANS</span>
            </div>
            <div style="color: #cbd5e1; font-size: 0.95rem; margin-top: 0.4rem;">
                K-Means achieved the highest Silhouette Score (<b>0.3592</b>), lowest Davies-Bouldin Index (<b>1.2597</b>), 
                and highest Calinski-Harabasz Score (<b>1,471.25</b>), clearly separating attacking and defensive player archetypes.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4 Model Cards
    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown(
            f"""
            <div class="model-card best">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="color: #fff; margin: 0;">1. K-Means Clustering</h3>
                        <span class="model-badge winner">Best Model</span>
                    </div>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">
                        Centroid-based iterative partitioning algorithm. Minimizes within-cluster sum of squares (inertia) 
                        across the 39 standardized statistical dimensions. Optimal K=2 selected via silhouette score maximization.
                    </p>
                </div>
                <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.1);">
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.3rem;">
                        <span style="color: #94a3b8;">Silhouette Score:</span>
                        <span style="color: #00e676; font-weight: 700;">{models_res['metrics']['K-Means']['silhouette']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.3rem;">
                        <span style="color: #94a3b8;">Davies-Bouldin Index:</span>
                        <span style="color: #00e676; font-weight: 700;">{models_res['metrics']['K-Means']['davies_bouldin']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem;">
                        <span style="color: #94a3b8;">Calinski-Harabasz Score:</span>
                        <span style="color: #00e676; font-weight: 700;">{models_res['metrics']['K-Means']['calinski_harabasz']:.2f}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="model-card">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="color: #fff; margin: 0;">2. Agglomerative Clustering</h3>
                        <span class="model-badge" style="background: rgba(56, 189, 248, 0.2); color: #38bdf8;">Hierarchical</span>
                    </div>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">
                        Bottom-up hierarchical clustering with Ward's linkage. Merges pairs of clusters that minimize the 
                        increase in total within-cluster variance. Demonstrates near-equivalent performance to K-Means.
                    </p>
                </div>
                <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.1);">
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.3rem;">
                        <span style="color: #94a3b8;">Silhouette Score:</span>
                        <span style="color: #38bdf8; font-weight: 700;">{models_res['metrics']['Agglomerative']['silhouette']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.3rem;">
                        <span style="color: #94a3b8;">Davies-Bouldin Index:</span>
                        <span style="color: #38bdf8; font-weight: 700;">{models_res['metrics']['Agglomerative']['davies_bouldin']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem;">
                        <span style="color: #94a3b8;">Calinski-Harabasz Score:</span>
                        <span style="color: #38bdf8; font-weight: 700;">{models_res['metrics']['Agglomerative']['calinski_harabasz']:.2f}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m_col2:
        st.markdown(
            f"""
            <div class="model-card">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="color: #fff; margin: 0;">3. DBSCAN</h3>
                        <span class="model-badge" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b;">Density-Based</span>
                    </div>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">
                        Density-Based Spatial Clustering of Applications with Noise (eps=1.5, min_samples=10). 
                        Identifies dense statistical cores while marking isolated outlier profiles as noise (-1).
                    </p>
                </div>
                <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.1);">
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.3rem;">
                        <span style="color: #94a3b8;">Silhouette Score:</span>
                        <span style="color: #f59e0b; font-weight: 700;">{models_res['metrics']['DBSCAN']['silhouette']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.3rem;">
                        <span style="color: #94a3b8;">Davies-Bouldin Index:</span>
                        <span style="color: #f59e0b; font-weight: 700;">{models_res['metrics']['DBSCAN']['davies_bouldin']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem;">
                        <span style="color: #94a3b8;">Calinski-Harabasz Score:</span>
                        <span style="color: #f59e0b; font-weight: 700;">{models_res['metrics']['DBSCAN']['calinski_harabasz']:.2f}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="model-card">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="color: #fff; margin: 0;">4. Gaussian Mixture Model</h3>
                        <span class="model-badge" style="background: rgba(139, 92, 246, 0.2); color: #c4b5fd;">Probabilistic</span>
                    </div>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">
                        Soft clustering with Expectation-Maximization (EM). Assumes player statistics are generated from 
                        a mixture of Gaussian distributions with full covariance matrices. Components = 2.
                    </p>
                </div>
                <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.1);">
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.3rem;">
                        <span style="color: #94a3b8;">Silhouette Score:</span>
                        <span style="color: #c4b5fd; font-weight: 700;">{models_res['metrics']['Gaussian Mixture']['silhouette']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.3rem;">
                        <span style="color: #94a3b8;">Davies-Bouldin Index:</span>
                        <span style="color: #c4b5fd; font-weight: 700;">{models_res['metrics']['Gaussian Mixture']['davies_bouldin']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.88rem;">
                        <span style="color: #94a3b8;">Calinski-Harabasz Score:</span>
                        <span style="color: #c4b5fd; font-weight: 700;">{models_res['metrics']['Gaussian Mixture']['calinski_harabasz']:.2f}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Optimal K Tuning Analysis
    st.markdown("---")
    st.markdown("### 🔍 Hyperparameter Tuning: K Selection via Silhouette & Elbow Methods")
    st.caption("Demonstrating why K=2 was scientifically selected as the optimal cluster count:")

    k_metrics = get_k_analysis(data["X_scaled"])
    fig_elbow = plot_elbow_and_silhouette(k_metrics)
    st.plotly_chart(fig_elbow, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 5: 📈 MODEL COMPARISON
# -----------------------------------------------------------------------------
elif selected_page == "📈 Model Comparison":
    st.markdown('<div class="main-header">MODEL BENCHMARK & COMPARISON</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Quantitative comparison of the 4 clustering models across three statistical evaluation standards</div>', unsafe_allow_html=True)

    # Comparison Table
    st.markdown("#### 📋 Comprehensive Model Evaluation Table")
    comp_df = models_res["comparison_df"].copy()
    comp_df = comp_df.sort_values(by="Silhouette Score", ascending=False).reset_index(drop=True)
    st.dataframe(
        comp_df.style.highlight_max(
            subset=["Silhouette Score", "Calinski-Harabasz Score"], color="#064e3b"
        ).highlight_min(
            subset=["Davies-Bouldin Index"], color="#064e3b"
        ),
        use_container_width=True
    )

    # 3 Charts
    st.markdown("---")
    st.markdown("#### 📊 Metric-by-Metric Visual Benchmarking")

    c1, c2, c3 = st.columns(3)
    with c1:
        fig_sil = plot_model_metric_bar(comp_df, "Silhouette Score", higher_is_better=True)
        st.plotly_chart(fig_sil, use_container_width=True)
    with c2:
        fig_db = plot_model_metric_bar(comp_df, "Davies-Bouldin Index", higher_is_better=False)
        st.plotly_chart(fig_db, use_container_width=True)
    with c3:
        fig_ch = plot_model_metric_bar(comp_df, "Calinski-Harabasz Score", higher_is_better=True)
        st.plotly_chart(fig_ch, use_container_width=True)

    # Metric Explanations Card
    st.markdown("---")
    st.markdown("### 📖 Cluster Validation Metrics Explained")

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 1.1rem; height: 100%;">
                <h4 style="color: #00e676; margin-top: 0;">Silhouette Score (↑)</h4>
                <p style="color: #94a3b8; font-size: 0.88rem;">
                    Measures how well each player fits into their assigned cluster versus neighboring clusters. 
                    Ranges from <b>-1 to +1</b>. Higher values indicate tight, well-separated clusters.
                </p>
                <div style="color: #e2e8f0; font-size: 0.85rem; font-weight: 600;">Leader: K-Means (0.3592)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_e2:
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 1.1rem; height: 100%;">
                <h4 style="color: #38bdf8; margin-top: 0;">Davies-Bouldin Index (↓)</h4>
                <p style="color: #94a3b8; font-size: 0.88rem;">
                    Measures average similarity between each cluster and its most similar counterpart. 
                    <b>Lower is better</b> (0 is optimal). Penalizes high intra-cluster dispersion.
                </p>
                <div style="color: #e2e8f0; font-size: 0.85rem; font-weight: 600;">Leader: K-Means (1.2597)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_e3:
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 1.1rem; height: 100%;">
                <h4 style="color: #c4b5fd; margin-top: 0;">Calinski-Harabasz Score (↑)</h4>
                <p style="color: #94a3b8; font-size: 0.88rem;">
                    Variance ratio criterion measuring ratio of between-cluster variance to within-cluster variance. 
                    <b>Higher values</b> signify defined and separated clusters.
                </p>
                <div style="color: #e2e8f0; font-size: 0.85rem; font-weight: 600;">Leader: K-Means (1,471.25)</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# PAGE 6: 🔬 PCA VISUALIZATION
# -----------------------------------------------------------------------------
elif selected_page == "🔬 PCA Visualization":
    st.markdown('<div class="main-header">PRINCIPAL COMPONENT ANALYSIS (PCA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">2-Dimensional Projection of the 39-Dimensional Feature Space</div>', unsafe_allow_html=True)

    ev = data["explained_variance"]

    # Variance Explained KPI Cards
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Principal Component 1 (PC1)</div>
                <div class="kpi-value">{ev['PC1_pct']}</div>
                <span class="kpi-badge">Primary variance driver</span>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Principal Component 2 (PC2)</div>
                <div class="kpi-value">{ev['PC2_pct']}</div>
                <span class="kpi-badge">Secondary orthogonal axis</span>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Explained Variance</div>
                <div class="kpi-value" style="color: #00e676;">{ev['Total_pct']}</div>
                <span class="kpi-badge">Combined 2D Information</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-callout">
            <b>Technical Clarification:</b> PCA is used exclusively for <b>dimensionality reduction and visualization</b>, 
            condensing 68.38% of total statistical variance into a human-interpretable 2D plane. 
            All machine learning models and similarity calculations operate directly on the <b>full 39-dimensional scaled space</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    # PCA Interactive Visual Controls
    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        model_choice = st.selectbox(
            "Color Scatter by Model Labels:",
            options=["K-Means", "Agglomerative", "Gaussian Mixture", "DBSCAN"],
            index=0
        )
    with ctrl_col2:
        all_players_pca = ["None"] + sorted(data["player_info"]["Player"].unique().tolist())
        target_highlight = st.selectbox(
            "Highlight a Specific Player on the 2D Plane:",
            options=all_players_pca,
            index=0
        )

    chosen_labels = models_res["labels"][model_choice]
    highlight_name = None if target_highlight == "None" else target_highlight

    fig_pca = plot_pca_clusters(
        data["pca_df"],
        data["player_info"],
        chosen_labels,
        model_name=model_choice,
        highlight_player=highlight_name
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # PCA Component Loadings Interpretation
    with st.expander("🔬 Principal Component Interpretation & Loadings"):
        pca_model = data["pca"]
        feat_names = data["performance_features"]
        loadings_df = pd.DataFrame(
            pca_model.components_.T,
            columns=["PC1 Loading", "PC2 Loading"],
            index=feat_names
        )
        c_load1, c_load2 = st.columns(2)
        with c_load1:
            st.markdown("**Top Positive Contributors to PC1 (Volume & Involvement):**")
            st.dataframe(loadings_df.sort_values(by="PC1 Loading", ascending=False).head(8)[["PC1 Loading"]], use_container_width=True)
        with c_load2:
            st.markdown("**Top Contributors to PC2 (Attacking vs. Defending Divergence):**")
            st.dataframe(loadings_df.sort_values(by="PC2 Loading", ascending=False).head(8)[["PC2 Loading"]], use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 7: 📋 DATASET EXPLORER
# -----------------------------------------------------------------------------
elif selected_page == "📋 Dataset Explorer":
    st.markdown('<div class="main-header">DATASET EXPLORER & FILTERING ENGINE</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Inspect raw records, player-level aggregated statistics, and custom scouting filters</div>', unsafe_allow_html=True)

    # Summary Badges
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Raw Dataset Rows", f"{data['raw_rows']:,}")
    b2.metric("Aggregated Unique Players", f"{data['unique_players']:,}")
    b3.metric("Performance Features", f"{data['feature_count']}")
    b4.metric("K-Means Clusters", f"{len(set(models_res['best_labels']))}")

    st.markdown("---")

    # View Mode Toggle
    view_mode = st.radio(
        "Select Dataset View:",
        ["Player-Level Cleaned Dataset (Aggregated 2,702 Players)", "Raw Imported Dataset (2,854 Records)"],
        horizontal=True
    )

    if "Player-Level" in view_mode:
        # Build exploration dataframe without column duplication
        info_cols = [c for c in data["player_info"].columns if c not in data["player_features"].columns or c == "Player"]
        merged_explorer = data["player_info"][info_cols].merge(data["player_features"], on="Player")

        # Filtering controls
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            search_query = st.text_input("🔍 Search Player Name:", "")
        with f_col2:
            pos_options = ["All"] + sorted(merged_explorer["Pos"].dropna().unique().tolist())
            selected_pos = st.selectbox("Position Filter:", pos_options)
        with f_col3:
            squad_options = ["All"] + sorted(merged_explorer["Squad"].dropna().unique().tolist())
            selected_squad = st.selectbox("Squad / Club Filter:", squad_options)
        with f_col4:
            min_minutes = st.slider("Minimum Minutes Played:", min_value=0, max_value=int(merged_explorer["Min"].max()), value=90, step=90)

        # Apply filters
        filtered_df = merged_explorer.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df["Player"].str.contains(search_query, case=False, na=False)]
        if selected_pos != "All":
            filtered_df = filtered_df[filtered_df["Pos"] == selected_pos]
        if selected_squad != "All":
            filtered_df = filtered_df[filtered_df["Squad"] == selected_squad]
        filtered_df = filtered_df[filtered_df["Min"] >= min_minutes]

        st.markdown(f"**Showing {len(filtered_df):,} matching players:**")
        st.dataframe(filtered_df, use_container_width=True)

        # Download CSV button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name="filtered_player_analytics.csv",
            mime="text/csv"
        )
    else:
        st.caption("Raw records as loaded directly from players_data-2024_2025.csv before cleaning and aggregation:")
        st.dataframe(data["raw_df"].head(100), use_container_width=True)
        st.info("Showing first 100 raw records for performance.")


# -----------------------------------------------------------------------------
# PAGE 8: ℹ️ ABOUT PROJECT
# -----------------------------------------------------------------------------
elif selected_page == "ℹ️ About Project":
    st.markdown('<div class="main-header">ABOUT THE PROJECT</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Academic & Machine Learning Documentation</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background: #111827; border: 1px solid #374151; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <h3 style="color: #ffffff; margin-top: 0;">Case Study 108: Player Similarity Analysis Using Machine Learning</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
                <b>Problem Statement:</b> In professional sports management, clubs face constant challenges in identifying recruitment targets, 
                benchmarking squad profiles, and finding tactical replacements for departing or injured players. 
                This project develops an objective, data-driven recommendation engine using unsupervised machine learning and vector similarity 
                to quantify player comparability across complex, multi-dimensional metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("### 🎯 Core Objectives")
        st.markdown(
            """
            - **End-to-End ML Pipeline:** Clean, impute, aggregate, scale, and model raw multi-match football telemetry.
            - **Player-Level Data Aggregation:** Successfully resolve multi-club and multi-league records into unified player profiles.
            - **Unsupervised Model Benchmarking:** Implement and compare K-Means, Agglomerative Clustering, DBSCAN, and Gaussian Mixture Models.
            - **Model Selection:** Rigorously validate clusters using Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.
            - **Similarity Recommendation:** Deploy high-precision Cosine Similarity across 39 standardized statistical features to surface Top 5 player alternatives.
            - **Interactive Analytics UI:** Deliver a production-grade sports analytics dashboard for faculty review and live scouting.
            """
        )

        st.markdown("### 🛠️ Technology Stack")
        st.markdown(
            """
            - **Language:** Python 3.12
            - **Data Manipulation:** Pandas, NumPy
            - **Machine Learning:** Scikit-learn (KMeans, Agglomerative, DBSCAN, GaussianMixture, PCA, StandardScaler, Cosine Similarity)
            - **Visualizations:** Plotly Express & Graph Objects, Matplotlib, Seaborn
            - **Web Application:** Streamlit
            """
        )

    with col_a2:
        st.markdown("### 📊 Project Methodology & Results Summary")
        st.markdown(
            f"""
            | Parameter | Project Value |
            | :--- | :--- |
            | **Raw Rows** | {data['raw_rows']:,} |
            | **Unique Players** | {data['unique_players']:,} |
            | **Performance Dimensions** | {data['feature_count']} features |
            | **Scaler** | StandardScaler (Zero-mean, unit-variance) |
            | **PCA Variance (PC1 + PC2)** | 68.38% (49.36% + 19.01%) |
            | **Clustering Algorithms** | 4 (K-Means, Agglomerative, DBSCAN, GMM) |
            | **Optimal Cluster Count** | K = 2 (Silhouette Score: 0.3592) |
            | **Winning Algorithm** | 🏆 **K-Means Clustering** |
            | **Similarity Metric** | Cosine Similarity on scaled vectors |
            | **Recommendation Output** | Top 5 Statistically Closest Players |
            """
        )

        st.markdown("### 🎓 Academic Viva Reference Guide")
        st.markdown(
            """
            - **Why unsupervised learning?** Player similarity does not have pre-labeled ground truth; algorithms must discover natural geometric groupings in performance space.
            - **Why Cosine Similarity over Euclidean Distance?** Cosine similarity focuses on the **orientation and proportion** of player profiles rather than raw volume alone, making it ideal for comparing tactical play styles.
            - **Why K=2?** Silhouette analysis across K=2 to 10 peaks at K=2 (0.3592), naturally splitting offensive/playmaking players and defensive/possession players.
            """
        )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 1rem 0;">
        ⚽ <b>Player Similarity Analysis</b> • Machine Learning Major Project • Case Study 108 • Built with Streamlit & Scikit-Learn
    </div>
    """,
    unsafe_allow_html=True
)
