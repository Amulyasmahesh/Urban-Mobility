"""
Urban Mobility — ML Pipeline
Model 1: Surge Trip Classifier (Random Forest + XGBoost)
Model 2: Demand Hotspot Clustering (KMeans)
Model 3: Fare Regression (XGBoost)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection  import train_test_split
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import (classification_report, confusion_matrix,
                                      roc_auc_score, mean_absolute_error, r2_score)
from sklearn.cluster          import KMeans

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not found — using GradientBoosting instead")
    XGB_AVAILABLE = False

# ── Load Data ────────────────────────────────────────────────
print("=" * 60)
print("0. Loading cleaned dataset")
print("=" * 60)
df = pd.read_csv("data/trips_clean.csv")
print(f"Shape: {df.shape}")


# ════════════════════════════════════════════════════════════
# MODEL 1 — Surge Trip Classifier
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 1 — Surge Trip Classifier")
print("=" * 60)

le = LabelEncoder()
df["zone_type_enc"] = le.fit_transform(df["zone_type"])

CLF_FEATURES = [
    "hour", "weekday", "is_weekend", "is_rush_hour", "is_late_night",
    "month", "pickup_zone_id", "dropoff_zone_id",
    "distance_km", "passenger_count",
    "zone_demand_score", "avg_speed_kmh", "zone_type_enc",
]

X = df[CLF_FEATURES].fillna(0)
y = df["is_surge_trip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"Surge rate: {y_train.mean()*100:.1f}%")

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                             min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("\n── Random Forest ──")
print(classification_report(y_test, rf_preds, target_names=["Normal", "Surge"]))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")

# XGBoost
if XGB_AVAILABLE:
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="logloss", random_state=42, verbosity=0)
else:
    from sklearn.ensemble import GradientBoostingClassifier
    xgb = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)

xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

print("\n── XGBoost ──")
print(classification_report(y_test, xgb_preds, target_names=["Normal", "Surge"]))
print(f"ROC-AUC: {roc_auc_score(y_test, xgb_proba):.4f}")

# Feature Importance
feat_imp = pd.Series(rf.feature_importances_, index=CLF_FEATURES).sort_values(ascending=False)
print("\nTop 10 features:\n", feat_imp.head(10))

fig, ax = plt.subplots(figsize=(10, 5))
feat_imp.head(12).plot.barh(ax=ax, color="#4F86C6")
ax.set_title("Feature Importance — Surge Classifier")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()
print("✅  Feature importance → outputs/feature_importance.png")

# Confusion Matrix
cm_arr = confusion_matrix(y_test, xgb_preds)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm_arr, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Normal","Surge"]); ax.set_yticklabels(["Normal","Surge"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — XGBoost")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm_arr[i,j], ha="center", va="center", fontsize=14,
                color="white" if cm_arr[i,j] > cm_arr.max()/2 else "black")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()
print("✅  Confusion matrix → outputs/confusion_matrix.png")


# ════════════════════════════════════════════════════════════
# MODEL 2 — Demand Hotspot Clustering (KMeans)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 2 — Demand Hotspot Clustering")
print("=" * 60)

ZONE_COORDS = {
    1:(40.7128,-74.0060), 2:(40.7549,-73.9840), 3:(40.7736,-73.9566),
    4:(40.6960,-73.9937), 5:(40.6413,-73.7781), 6:(40.7769,-73.8740),
    7:(40.7081,-73.9571), 8:(40.8116,-73.9465),}