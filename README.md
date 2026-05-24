# 🩺 Global Diabetes Burden Clustering

> *Which countries are sitting on an untreated diabetes crisis?*

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Data](https://img.shields.io/badge/Data-WHO%20%7C%20NCD--RisC%20%7C%20World%20Bank-green)

---

## 🔴 The Problem

Over **50% of diabetics globally are undiagnosed and untreated** — mostly in countries that can't afford to manage the disease. No one has clearly mapped *which* countries are in the most dangerous position: high diabetes burden combined with low healthcare capacity.

This project uses **unsupervised machine learning** to answer that question — with no predefined labels, letting the data discover natural country groupings itself.

---

## 🎯 Project Goals

- Cluster 124 countries by their diabetes burden and healthcare capacity
- Discover which nations face the highest diabetes burden with the least ability to manage it
- Identify anomaly countries that break expected patterns
- Deploy findings as an interactive web dashboard

---

## 📦 Datasets Used

| # | Dataset | Source | Feature |
|---|---|---|---|
| 1 | Diabetes Prevalence (%) | NCD-RisC (2022) | `diabetes_prevalence` |
| 2 | Health Spending per Person (PPP) | World Bank via OWID | `health_spending` |
| 3 | Doctors per 1,000 People | WHO via OWID | `doctors_per_1000` |
| 4 | Out-of-Pocket Spending (%) | WHO via OWID | `oop_spending` |

- **Year:** 2022
- **Countries:** 124 (after merging all 4 datasets)
- **Missing values:** 0

> Note: Health spending is PPP-adjusted (international $) to ensure fair comparison across countries with different price levels and cost of living.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core language |
| **Pandas / NumPy** | Data loading, merging, cleaning |
| **Scikit-learn** | StandardScaler, K-Means, PCA, Isolation Forest |
| **Matplotlib / Seaborn** | EDA visualizations |
| **Plotly** | Interactive world map |
| **Streamlit** | Web app deployment |
| **Google Colab** | Development environment |

---

## 🧠 Machine Learning Pipeline
