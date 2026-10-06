# Player Similarity Analysis - Streamlit Application

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://player-similarity-analysis-5qk6oraobr77tvvg6lvzni.streamlit.app/)

> 🚀 **Live Application:** [https://player-similarity-analysis-5qk6oraobr77tvvg6lvzni.streamlit.app/](https://player-similarity-analysis-5qk6oraobr77tvvg6lvzni.streamlit.app/)

A sports analytics web application and interactive player recommendation engine built for **Machine Learning Major Project — Case Study 108: Player Similarity Analysis**.

---

## Overview

In modern football recruitment and team building, sports organizations need objective, data-driven systems to identify players with comparable performance profiles, scout emerging talent, and find tactical replacements for key squad members.

This application uses **unsupervised machine learning** and **cosine similarity** to analyze multi-dimensional football performance metrics. It provides an intuitive, interactive scouting dashboard where users can inspect player statistics, visualize high-dimensional clusters via PCA, compare clustering models, and generate the **Top 5 statistically similar players** for any chosen athlete.

---

## Features

- **Player Recommendation Engine:** Select any of the 2,702 unique players to inspect their profile and compute their Top 5 most similar peers using Cosine Similarity.
- **Radar & Bar Profile Comparisons:** Visualize player strengths and tactical styles against recommended peers across attacking, playmaking, and defensive metrics.
- **Exploratory Player Analytics:** Interactive histograms, distribution curves, goal/assist leaderboards, and a Pearson correlation heatmap.
- **Clustering Model Evaluation:** In-depth evaluation of 4 unsupervised algorithms: K-Means, Agglomerative Hierarchical Clustering, DBSCAN, and Gaussian Mixture Models (GMM).
- **Model Comparison Benchmarks:** Side-by-side performance comparisons across three standard cluster validation metrics:
  - **Silhouette Score** (Higher is better)
  - **Davies-Bouldin Index** (Lower is better)
  - **Calinski-Harabasz Score** (Higher is better)
- **2D PCA Visualization:** Principal Component Analysis reducing 39 features to PC1 and PC2 (explaining **68.38%** of total variance) with interactive cluster coloring and player location highlighting.
- **Dataset Explorer:** Filter and search across 2,702 players by position, squad, age, and playing time, with one-click CSV export.

---

## Machine Learning Approach

The application reproduces the exact ML workflow established in `PlayerClassification.ipynb`:

```
Raw Football Player Dataset (2,854 records)
                 ↓
      Data Understanding & EDA
                 ↓
           Data Cleaning
                 ↓
  Player-Level Aggregation (2,702 unique players)
                 ↓
Feature Selection (39 performance features)
                 ↓
    Median Missing Value Imputation
                 ↓
     StandardScaler Normalization
                 ↓
  PCA (Dimensionality Reduction: 68.38% Variance)
                 ↓
 4 Clustering Models Evaluated (Optimal K = 2)
                 ↓
 Model Benchmarking (Best Model: K-Means)
                 ↓
 Pairwise Cosine Similarity (2,702 × 2,702)
                 ↓
  Top 5 Similar Player Recommendations
```

### Model Performance Summary

| Model | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **K-Means** | **0.3592** | **1.2597** | **1,471.25** | 🏆 **Best Model** |
| **Agglomerative Clustering** | 0.3572 | 1.3293 | 1,290.53 | Benchmark |
| **DBSCAN** | 0.1372 | 1.3413 | 62.08 | Density Baseline |
| **Gaussian Mixture Model** | 0.2389 | 1.3921 | 1,069.55 | Probabilistic |

---

## Dataset

- **File:** `data/players_data-2024_2025.csv`
- **Raw Records:** 2,854 entries
- **Unique Players (after aggregation):** 2,702 players
- **Feature Dimensions:** 39 key performance indicators spanning goals, assists, expected metrics (xG, xAG), passing volume/accuracy, progression, defensive actions (tackles, interceptions, clearances), and aerial duels.

---

## Project Structure

```
Streamlit/
│
├── app.py                      # Main Streamlit application and page router
├── requirements.txt            # Python dependencies
├── README.md                   # Application documentation
│
├── data/
│   └── players_data-2024_2025.csv  # Player performance dataset
│
├── models/
│   └── clustering_artifacts.joblib # Saved model instances and evaluation cache
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py        # Data cleaning, aggregation, scaling, and PCA
│   ├── similarity.py           # Cosine similarity matrix and recommendation logic
│   └── visualization.py        # Plotly charts styled with dark sports theme
│
└── assets/
    └── style.css               # Modern dark sports analytics stylesheet
```

---

## Installation

Ensure Python 3.10+ is installed. Then install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Application

From the `Streamlit` folder:

```bash
streamlit run app.py
```

Or from the root project directory:

```bash
streamlit run Streamlit/app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## Application Pages

1. **🏠 Dashboard:** High-level project KPIs, workflow diagram, position distribution, and featured star players.
2. **👤 Player Similarity:** Core recommendation engine: select any player to inspect their statistical profile, discover their Top 5 similar counterparts with cosine similarity percentages, and explore radar and bar comparison charts.
3. **📊 Player Analytics:** Distributions for goals, assists, age, and minutes, plus top scorer/assist leaderboards and correlation heatmaps.
4. **🧠 ML Models:** Technical breakdown of K-Means, Agglomerative Clustering, DBSCAN, and GMM, including hyperparameter tuning (Elbow & Silhouette curves).
5. **📈 Model Comparison:** Quantitative evaluation across Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.
6. **🔬 PCA Visualization:** 2D interactive scatter plot of PC1 vs. PC2 with cluster coloring and individual player locator.
7. **📋 Dataset Explorer:** Filterable, sortable player table with custom scouting filters and CSV export.
8. **ℹ️ About Project:** Academic problem statement, methodology, technology stack, and viva discussion guide.
