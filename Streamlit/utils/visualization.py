"""
Interactive Visualization Utilities for Football Player Analytics Dashboard.

Provides high-performance Plotly interactive charts styled with a dark
sports analytics aesthetic (deep slate, pitch green, electric blue accents).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.preprocessing import CORRELATION_FEATURES
from utils.similarity import RADAR_METRICS

# Color Palette Constants
THEME = {
    "bg": "#0e1117",
    "card_bg": "#161b22",
    "border": "#30363d",
    "text": "#e2e8f0",
    "text_muted": "#94a3b8",
    "pitch_green": "#10b981",
    "bright_green": "#00e676",
    "cyber_blue": "#38bdf8",
    "deep_blue": "#2563eb",
    "purple": "#8b5cf6",
    "amber": "#f59e0b",
    "crimson": "#f43f5e",
    "cluster_colors": ["#10b981", "#38bdf8", "#8b5cf6", "#f59e0b", "#f43f5e", "#ec4899", "#94a3b8"],
}


def apply_dark_theme(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Applies unified dark sports analytics layout styling to any Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 16, "color": THEME["text"], "family": "Inter, sans-serif"},
            "x": 0.02,
            "xanchor": "left",
        },
        paper_bgcolor=THEME["card_bg"],
        plot_bgcolor=THEME["card_bg"],
        font={"color": THEME["text"], "family": "Inter, sans-serif"},
        height=height,
        margin={"l": 40, "r": 30, "t": 50 if title else 25, "b": 40},
        xaxis={
            "gridcolor": "rgba(255, 255, 255, 0.07)",
            "zerolinecolor": "rgba(255, 255, 255, 0.12)",
            "tickfont": {"color": THEME["text_muted"]},
            "title": {"font": {"color": THEME["text"]}},
        },
        yaxis={
            "gridcolor": "rgba(255, 255, 255, 0.07)",
            "zerolinecolor": "rgba(255, 255, 255, 0.12)",
            "tickfont": {"color": THEME["text_muted"]},
            "title": {"font": {"color": THEME["text"]}},
        },
        legend={
            "bgcolor": "rgba(22, 27, 34, 0.8)",
            "bordercolor": THEME["border"],
            "font": {"color": THEME["text"]},
        },
        hoverlabel={
            "bgcolor": "#1e293b",
            "font": {"color": "#f8fafc", "size": 13},
            "bordercolor": THEME["pitch_green"],
        },
    )
    return fig


def plot_position_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of player distribution by position (Pos)."""
    pos_counts = df["Pos"].value_counts().reset_index()
    pos_counts.columns = ["Position", "Count"]
    pos_counts["Percentage"] = ((pos_counts["Count"] / len(df)) * 100).round(1)

    fig = go.Figure(
        data=[
            go.Bar(
                x=pos_counts["Position"],
                y=pos_counts["Count"],
                text=pos_counts["Count"],
                textposition="auto",
                customdata=pos_counts["Percentage"],
                hovertemplate="<b>Position:</b> %{x}<br><b>Count:</b> %{y}<br><b>Share:</b> %{customdata}%<extra></extra>",
                marker={
                    "color": pos_counts["Count"],
                    "colorscale": [[0, "#047857"], [1, THEME["bright_green"]]],
                    "line": {"color": THEME["border"], "width": 1},
                },
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Playing Position",
        yaxis_title="Number of Players",
    )
    return apply_dark_theme(fig, "Player Position Distribution")


def plot_metric_distribution(df: pd.DataFrame, column: str, title: str, color: str = THEME["pitch_green"], bins: int = 25) -> go.Figure:
    """Interactive histogram for goals, assists, age, or minutes played."""
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    mean_val = float(series.mean())
    median_val = float(series.median())

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=series,
            nbinsx=bins,
            marker={"color": color, "line": {"color": "#064e3b", "width": 1}, "opacity": 0.85},
            hovertemplate=f"<b>Range:</b> %{{x}}<br><b>Count:</b> %{{y}}<extra></extra>",
            name=column,
        )
    )

    # Median line
    fig.add_vline(
        x=median_val,
        line_dash="dash",
        line_color=THEME["cyber_blue"],
        annotation_text=f"Median: {median_val:.1f}",
        annotation_position="top right",
        annotation_font_color=THEME["cyber_blue"],
    )

    fig.update_layout(
        xaxis_title=title,
        yaxis_title="Number of Players",
        showlegend=False,
    )
    return apply_dark_theme(fig, f"{title} Distribution")


def plot_top_players_metric(df: pd.DataFrame, metric_col: str, title: str, top_n: int = 15, color: str = THEME["pitch_green"]) -> go.Figure:
    """Horizontal bar chart for top goal scorers or assist leaders."""
    top_df = (
        df.groupby("Player")[metric_col]
        .sum()
        .sort_values(ascending=True)
        .tail(top_n)
        .reset_index()
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=top_df[metric_col],
                y=top_df["Player"],
                orientation="h",
                text=top_df[metric_col],
                textposition="auto",
                marker={
                    "color": top_df[metric_col],
                    "colorscale": [[0, "#0284c7"], [1, color]],
                    "line": {"color": THEME["border"], "width": 1},
                },
                hovertemplate="<b>%{y}</b><br>" + f"{metric_col}: " + "<b>%{x}</b><extra></extra>",
            )
        ]
    )
    fig.update_layout(
        xaxis_title=metric_col,
        yaxis_title="Player",
        height=480,
    )
    return apply_dark_theme(fig, title, height=480)


def plot_goals_vs_assists(df: pd.DataFrame) -> go.Figure:
    """Goals vs Assists scatter plot with player hover."""
    sample = df[["Player", "Squad", "Pos", "Gls", "Ast", "Min"]].copy()
    sample["Gls"] = pd.to_numeric(sample["Gls"], errors="coerce").fillna(0)
    sample["Ast"] = pd.to_numeric(sample["Ast"], errors="coerce").fillna(0)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=sample["Gls"],
                y=sample["Ast"],
                mode="markers",
                marker={
                    "size": 7,
                    "color": sample["Gls"] + sample["Ast"],
                    "colorscale": "Viridis",
                    "opacity": 0.75,
                    "line": {"color": "rgba(255,255,255,0.2)", "width": 0.5},
                    "colorbar": {"title": "G+A"},
                },
                customdata=np.stack([sample["Player"], sample["Squad"], sample["Pos"]], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>Pos: %{customdata[2]}<br>Goals: %{x}<br>Assists: %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Goals Scored (Gls)",
        yaxis_title="Assists Provided (Ast)",
    )
    return apply_dark_theme(fig, "Goals vs. Assists Correlation")


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Interactive correlation matrix heatmap for selected key performance indicators."""
    valid_cols = [c for c in CORRELATION_FEATURES if c in df.columns]
    corr_df = df[valid_cols].apply(pd.to_numeric, errors="coerce").corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar={"title": "Pearson r"},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={"tickangle": -45},
        height=620,
    )
    return apply_dark_theme(fig, "Selected Performance Features Correlation Heatmap", height=620)


def plot_elbow_and_silhouette(k_metrics: dict) -> go.Figure:
    """Dual-axis plot showing Inertia (Elbow) and Silhouette Score for K in [2, 10]."""
    k_list = k_metrics["k_list"]
    inertias = k_metrics["inertias"]
    silhouettes = k_metrics["silhouette_scores"]
    optimal_k = k_metrics["optimal_k"]

    fig = go.Figure()

    # Inertia line
    fig.add_trace(
        go.Scatter(
            x=k_list,
            y=inertias,
            mode="lines+markers",
            name="Inertia (Elbow)",
            line={"color": THEME["cyber_blue"], "width": 3},
            marker={"size": 8, "symbol": "circle"},
            yaxis="y1",
        )
    )

    # Silhouette line
    fig.add_trace(
        go.Scatter(
            x=k_list,
            y=silhouettes,
            mode="lines+markers",
            name="Silhouette Score",
            line={"color": THEME["bright_green"], "width": 3, "dash": "solid"},
            marker={"size": 10, "symbol": "diamond"},
            yaxis="y2",
        )
    )

    # Highlight Optimal K
    fig.add_vline(
        x=optimal_k,
        line_dash="dot",
        line_color=THEME["amber"],
        line_width=2,
        annotation_text=f"Optimal K = {optimal_k}",
        annotation_position="top left",
        annotation_font_color=THEME["amber"],
    )

    fig.update_layout(
        title={"text": "<b>K-Means Optimization: Elbow Method & Silhouette Analysis</b>"},
        xaxis={"title": "Number of Clusters (K)", "tickmode": "linear", "tick0": 2, "dtick": 1},
        yaxis={"title": {"text": "Inertia (Within-Cluster Sum of Squares)", "font": {"color": THEME["cyber_blue"]}}},
        yaxis2={
            "title": {"text": "Silhouette Score", "font": {"color": THEME["bright_green"]}},
            "overlaying": "y",
            "side": "right",
            "gridcolor": "rgba(0,0,0,0)",
        },
        legend={"x": 0.65, "y": 0.95},
    )
    return apply_dark_theme(fig, "K-Means Elbow & Silhouette Analysis", height=450)


def plot_model_metric_bar(comparison_df: pd.DataFrame, metric_col: str, higher_is_better: bool = True) -> go.Figure:
    """Bar chart comparing the 4 clustering models on a single evaluation metric."""
    df_sorted = comparison_df.copy()
    colors = []
    for model in df_sorted["Model"]:
        if model == "K-Means":
            colors.append(THEME["bright_green"])
        elif model == "Agglomerative":
            colors.append(THEME["cyber_blue"])
        elif model == "Gaussian Mixture":
            colors.append(THEME["purple"])
        else:
            colors.append(THEME["amber"])

    direction_note = "Higher is better ↑" if higher_is_better else "Lower is better ↓"

    fig = go.Figure(
        data=[
            go.Bar(
                x=df_sorted["Model"],
                y=df_sorted[metric_col],
                text=df_sorted[metric_col].apply(lambda v: f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"),
                textposition="auto",
                marker={"color": colors, "line": {"color": THEME["border"], "width": 1.5}},
                hovertemplate="<b>%{x}</b><br>" + f"{metric_col}: " + "<b>%{y}</b><extra></extra>",
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Clustering Model",
        yaxis_title=f"{metric_col} ({direction_note})",
    )
    return apply_dark_theme(fig, f"{metric_col} Comparison ({direction_note})")


def plot_pca_clusters(
    pca_df: pd.DataFrame,
    player_info: pd.DataFrame,
    cluster_labels: np.ndarray,
    model_name: str = "K-Means",
    highlight_player: str = None
) -> go.Figure:
    """
    Interactive 2D PCA scatter plot showing player clusters on PC1 vs PC2.
    Optionally highlights the currently selected player.
    """
    plot_df = pca_df.copy()
    plot_df["Player"] = player_info["Player"].values
    plot_df["Squad"] = player_info["Squad"].values
    plot_df["Pos"] = player_info["Pos"].values
    plot_df["Age"] = player_info["Age"].values
    plot_df["Cluster"] = [f"Cluster {c}" if c != -1 else "Noise (-1)" for c in cluster_labels]

    unique_clusters = sorted(plot_df["Cluster"].unique())
    color_map = {}
    for i, c in enumerate(unique_clusters):
        if "Noise" in c:
            color_map[c] = "#64748b"
        else:
            color_map[c] = THEME["cluster_colors"][i % len(THEME["cluster_colors"])]

    fig = go.Figure()

    for cluster_name in unique_clusters:
        sub = plot_df[plot_df["Cluster"] == cluster_name]
        fig.add_trace(
            go.Scatter(
                x=sub["PC1"],
                y=sub["PC2"],
                mode="markers",
                name=cluster_name,
                marker={
                    "size": 6,
                    "color": color_map[cluster_name],
                    "opacity": 0.65,
                },
                customdata=np.stack([sub["Player"], sub["Squad"], sub["Pos"], sub["Age"]], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Club: %{customdata[1]}<br>"
                    "Pos: %{customdata[2]} | Age: %{customdata[3]}<br>"
                    "PC1: %{x:.2f} | PC2: %{y:.2f}<extra></extra>"
                ),
            )
        )

    # Highlight selected player if provided
    if highlight_player and highlight_player in plot_df["Player"].values:
        p_row = plot_df[plot_df["Player"] == highlight_player].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[p_row["PC1"]],
                y=[p_row["PC2"]],
                mode="markers+text",
                name="Selected Player",
                text=[f"⭐ {highlight_player}"],
                textposition="top center",
                textfont={"size": 13, "color": "#ffffff"},
                marker={
                    "size": 15,
                    "color": "#fbbf24",
                    "symbol": "star",
                    "line": {"color": "#ffffff", "width": 2},
                },
                hovertemplate=f"<b>⭐ {highlight_player} (Selected)</b><br>Cluster: {p_row['Cluster']}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Principal Component 1 (49.36% Variance)",
        yaxis_title="Principal Component 2 (19.01% Variance)",
        height=550,
        legend={"title": "Clusters", "orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return apply_dark_theme(fig, f"{model_name} - Player Clusters in 2D PCA Space", height=550)


def plot_player_radar(radar_df: pd.DataFrame, selected_player: str, top_similar_player: str) -> go.Figure:
    """
    Radar (Spider) chart comparing the selected player with the top recommended similar player
    across normalized (0-100) performance categories.
    """
    categories = [m["name"] for m in RADAR_METRICS if m["col"] in radar_df.columns]
    feature_cols = [m["col"] for m in RADAR_METRICS if m["col"] in radar_df.columns]

    sel_row = radar_df[radar_df["Player"] == selected_player]
    top_row = radar_df[radar_df["Player"] == top_similar_player]

    if sel_row.empty:
        return go.Figure()

    sel_values = sel_row[feature_cols].iloc[0].values.tolist()
    # Close the radar loop
    sel_values.append(sel_values[0])

    fig = go.Figure()

    # Trace 1: Selected Player
    fig.add_trace(
        go.Scatterpolar(
            r=sel_values,
            theta=categories + [categories[0]],
            fill="toself",
            name=f"Selected: {selected_player}",
            line={"color": THEME["bright_green"], "width": 2.5},
            fillcolor="rgba(16, 185, 129, 0.25)",
        )
    )

    # Trace 2: Top Similar Player (if present)
    if not top_row.empty:
        top_values = top_row[feature_cols].iloc[0].values.tolist()
        top_values.append(top_values[0])
        fig.add_trace(
            go.Scatterpolar(
                r=top_values,
                theta=categories + [categories[0]],
                fill="toself",
                name=f"Similar: {top_similar_player}",
                line={"color": THEME["cyber_blue"], "width": 2.5},
                fillcolor="rgba(56, 189, 248, 0.2)",
            )
        )

    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "color": THEME["text_muted"],
                "gridcolor": "rgba(255, 255, 255, 0.1)",
            },
            "angularaxis": {
                "color": THEME["text"],
                "gridcolor": "rgba(255, 255, 255, 0.1)",
                "tickfont": {"size": 11},
            },
            "bgcolor": THEME["card_bg"],
        },
        height=500,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "center", "x": 0.5},
    )
    return apply_dark_theme(fig, f"Statistical Profile Comparison: {selected_player} vs. {top_similar_player}", height=520)


def plot_metric_comparison_bars(
    player_features: pd.DataFrame,
    selected_player: str,
    similar_names: list,
    selected_metrics: list = None
) -> go.Figure:
    """Grouped bar chart comparing raw stats between the selected player and similar players."""
    if not selected_metrics:
        selected_metrics = ["Gls", "Ast", "xG", "xAG", "PrgC", "PrgP", "Tkl", "Int"]

    players_to_plot = [selected_player] + [p for p in similar_names[:4] if p != selected_player]
    subset = player_features[player_features["Player"].isin(players_to_plot)].copy()

    fig = go.Figure()
    palette = [THEME["bright_green"], THEME["cyber_blue"], THEME["purple"], THEME["amber"], THEME["crimson"]]

    for i, player in enumerate(players_to_plot):
        p_row = subset[subset["Player"] == player]
        if p_row.empty:
            continue
        vals = [p_row[m].iloc[0] if m in p_row.columns else 0 for m in selected_metrics]
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Bar(
                name=f"{'⭐ ' if player == selected_player else ''}{player}",
                x=selected_metrics,
                y=vals,
                marker={"color": color},
                hovertemplate="<b>%{x}</b>: %{y:.2f}<extra>" + player + "</extra>",
            )
        )

    fig.update_layout(
        barmode="group",
        xaxis_title="Performance Metric",
        yaxis_title="Stat Value (Raw / Aggregated)",
        height=420,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return apply_dark_theme(fig, "Key Metrics Comparison (Selected vs. Similar Players)", height=420)
