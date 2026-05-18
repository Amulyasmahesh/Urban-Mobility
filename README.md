# 🚗 UrbanGuard — Ride-Share Surge Pricing Analyzer

> End-to-end data analytics project covering Pandas, SQL, Machine Learning,
> and Tableau — built around a real-world urban mobility problem.

---

## 🎯 Problem Statement

Ride-share companies lose drivers during peak demand and overcharge riders
during low-demand windows. This project builds a full analytics pipeline to:
- Identify **surge hotspot zones** by time of day
- Predict whether a trip will be a **high-surge trip** (≥1.5x)
- Forecast **fare amounts** before a trip starts
- Cluster city zones into **demand tiers** for driver deployment

---

## 🗂️ Project Structure

```
urban_mobility/
│
├── data/
│   ├── generate_data.py        ← Synthetic dataset generator (50K trips)
│   ├── trips_raw.csv           ← Raw generated data
│   ├── trips_clean.csv         ← Cleaned + feature-engineered data
│   ├── zone_summary.csv        ← Zone-level aggregates for Tableau
│   └── fare_predictions.csv   ← ML model output for Tableau
│
├── notebooks/
│   ├── 01_eda.py               ← Pandas EDA & feature engineering
│   └── 02_ml_pipeline.py      ← ML models (classifier + clustering + regression)
│
├── sql/
│   └── analysis_queries.sql   ← 8 business SQL queries (CTEs, window fns)
│
├── tableau_guide/
│   └── dashboard_blueprint.md ← Full 4-dashboard Tableau blueprint
│
└── outputs/
    ├── eda_surge_plots.png
    ├── feature_importance.png
    ├── confusion_matrix.png
    ├── demand_clusters.png
    └── fare_regression.png
```

---

## 🛠️ Tech Stack

| Layer          | Tool / Library                          |
|----------------|-----------------------------------------|
| Data Wrangling | Python, Pandas, NumPy                   |
| Database       | SQLite / PostgreSQL                     |
| ML             | Scikit-learn, XGBoost                   |
| Visualization  | Matplotlib, Seaborn, Tableau            |
| IDE            | Jupyter Notebook / VS Code              |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# 2. Generate dataset
cd data && python generate_data.py

# 3. Run EDA
cd .. && python notebooks/01_eda.py

# 4. Run ML pipeline
python notebooks/02_ml_pipeline.py

# 5. Load SQL queries
# Import trips_clean.csv into DB Browser for SQLite or pgAdmin
# Run sql/analysis_queries.sql

# 6. Tableau
# Open Tableau → Connect to trips_clean.csv + zone_summary.csv
# Follow tableau_guide/dashboard_blueprint.md
```

---

## 📊 Key Results

| Model                  | Metric        | Score   |
|------------------------|---------------|---------|
| Surge Classifier (RF)  | ROC-AUC       | ~0.91   |
| Surge Classifier (XGB) | ROC-AUC       | ~0.93   |
| Fare Regressor (XGB)   | R²            | ~0.94   |
| Fare Regressor (XGB)   | MAE           | ~$1.80  |
| KMeans Clustering      | Clusters      | 3 zones |

---

## 💡 Business Insights

1. **Business zones surge 2.2x on average** during morning (7–9am) and evening (5–7pm) rush hours
2. **Entertainment zones** have the highest late-night surge (avg 2.8x after 10pm)
3. **Airport zones** show predictable surge patterns that can be pre-communicated to drivers
4. **Card payments** generate 18% higher tips on average vs cash
5. **Top 3 corridors** (Downtown↔Midtown, Airport↔Downtown) account for 31% of total revenue

---

## 🗺️ Tableau Dashboard Previews

- **Dashboard 1:** Executive KPI overview — total trips, revenue, avg surge
- **Dashboard 2:** City surge heatmap — zone × hour surge intensity
- **Dashboard 3:** Time intelligence — 7-day trend, rush vs off-peak
- **Dashboard 4:** ML insights — fare prediction scatter, zone cluster map
- https://public.tableau.com/authoring/urbanmobility/UrbanGuardRideShareSurgeAnalyzer#1(tableau dashboard link).

---

## 📌 Skills Demonstrated

- ✅ Data generation & realistic synthetic modelling
- ✅ Pandas: cleaning, type casting, groupby, pivot tables, feature engineering
- ✅ SQL: aggregations, CTEs, window functions (RANK, moving average)
- ✅ ML: binary classification, regression, unsupervised clustering, model evaluation
- ✅ Tableau: multi-dashboard design, calculated fields, map visualisation, interactivity
- ✅ End-to-end storytelling: problem → data → insight → decision

---

*Built as a portfolio project to demonstrate full-stack data analytics capability.*
