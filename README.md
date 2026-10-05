# ⚽ Football Player Similarity Analysis & Recommendation Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning Major Project (**Case Study 108**) for scouting, talent identification, and tactical player replacement using unsupervised clustering and cosine similarity.

---

## 📌 Project Overview

In professional football recruitment, clubs require objective, data-driven systems to identify players with comparable athletic and tactical profiles, scout undervalued talents, and find seamless replacements for departing squad members.

This project delivers:
1. **Machine Learning Pipeline:** Preprocessing, exploratory data analysis, feature selection (39 core metrics), standardization, and 2D PCA dimensionality reduction.
2. **Unsupervised Clustering Benchmark:** Comparative analysis of **K-Means**, **Agglomerative Hierarchical Clustering**, **DBSCAN**, and **Gaussian Mixture Models (GMM)** evaluated with Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.
3. **Similarity Engine:** High-dimensional pairwise Cosine Similarity matrix (2,702 × 2,702) generating top 5 statistically matching player recommendations.
4. **Interactive Streamlit Web Application:** A dark-themed, dynamic scouting dashboard featuring radar comparisons, cluster projections, statistical leaderboards, and dataset search.

---

## 🏆 Model Benchmarking Summary

Evaluated on 2,702 unique professional players (2024–2025 season) across 39 performance metrics:

| Model | Clusters ($K$) | Silhouette Score ↑ | Davies-Bouldin Index ↓ | Calinski-Harabasz Score ↑ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means** | **2** | **0.3592** | **1.2597** | **1,471.25** | 🏆 **Best Model** |
| **Agglomerative Hierarchical** | 2 | 0.3572 | 1.3293 | 1,290.53 | Strong Alternative |
| **Gaussian Mixture Model (GMM)** | 2 | 0.2389 | 1.3921 | 1,069.55 | Probabilistic |
| **DBSCAN** | 2 (eps=1.5, min=10) | 0.1372 | 1.3413 | 62.08 | Density Baseline |

*K-Means achieved the highest cohesion, lowest intra-cluster dispersion, and highest between-cluster separation.*

---

## 📁 Repository Structure

```text
Player-Similarity-Analysis/
│
├── Streamlit/                         # Interactive Streamlit Web Application
│   ├── app.py                         # Application entrypoint & multi-page router
│   ├── requirements.txt               # App-specific dependencies
│   ├── README.md                      # Streamlit documentation
│   ├── assets/
│   │   └── style.css                  # Custom sports analytics stylesheet
│   ├── data/
│   │   └── players_data-2024_2025.csv # Pre-packaged dataset
│   ├── models/
│   │   └── clustering_artifacts.joblib# Serialized models & evaluation metrics
│   └── utils/
│       ├── preprocessing.py           # Pipeline, feature selection & PCA
│       ├── models.py                  # Clustering models & evaluation metrics
│       ├── similarity.py              # Cosine similarity recommendation logic
│       └── visualization.py           # Plotly charts (radar, PCA, distributions)
│
├── ColabNotebook/                     # Google Colab / Jupyter Research Notebook
│   ├── PlayerClassification.ipynb     # Complete data science & modeling notebook
│   └── requirements.txt               # Research environment requirements
│
├── Dataset/                           # Primary dataset source
│   └── players_data-2024_2025.csv     # Raw 2024-2025 player performance dataset
│
├── Documentation+Report/              # Project report & documentation
│   └── PlayerSimilarityAnalysis.pdf   # Complete academic major project report
│
├── requirements.txt                   # Root dependency file for cloud deployments
└── .gitignore                         # Git exclusion rules
```

---

## 🚀 Quickstart & Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Whatisthissam/Player-Similarity-Analysis.git
cd Player-Similarity-Analysis
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# or: .venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run Streamlit/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Deploy to Streamlit Community Cloud

Deploying this application online is free and takes less than 2 minutes:

1. **Sign in to Streamlit Cloud:** Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. **Create New App:** Click **"Create app"** / **"New app"**.
3. **Configure Repository:**
   - **Repository:** `Whatisthissam/Player-Similarity-Analysis`
   - **Branch:** `main`
   - **Main file path:** `Streamlit/app.py`
4. **Deploy:** Click **"Deploy!"**.
   - Streamlit Cloud will automatically detect dependencies from `requirements.txt` and launch your live application with a public shareable URL.

---

## 💻 Research & Experimentation (Colab Notebook)

The complete end-to-end data science experiments, statistical tests, elbow/silhouette curves, and evaluation outputs are documented in [ColabNotebook/PlayerClassification.ipynb](file:///Users/sameerrathod/Desktop/MachineLearning-MajorProject/ColabNotebook/PlayerClassification.ipynb).

You can also run it directly in Google Colab by uploading the notebook and connecting the dataset from `Dataset/players_data-2024_2025.csv`.

---

## 📄 Documentation & Report

The comprehensive formal report detailing the theoretical background, mathematical formulation of clustering algorithms, distance metrics, and recruitment case studies is available in:
- [Documentation+Report/PlayerSimilarityAnalysis.pdf](file:///Users/sameerrathod/Desktop/MachineLearning-MajorProject/Documentation+Report/PlayerSimilarityAnalysis.pdf)

---

## 📜 License
This project is developed for academic and educational purposes under the MIT License.
