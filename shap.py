"""
Urban Mobility — SHAP Values
Explains WHY the XGBoost model made each prediction
Most impressive ML addition for interviews!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from xgboost                 import XGBClassifier

try:
    import shap
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv("data/trips_clean.csv")

le = LabelEncoder()
df["zone_type_enc"] = le.fit_transform(df["zone_type"])

FEATURES = [
    "hour", "weekday", "is_weekend", "is_rush_hour", "is_late_night",
    "month", "pickup_zone_id", "dropoff_zone_id",
    "distance_km", "passenger_count",
    "zone_demand_score", "avg_speed_kmh", "zone_type_enc",
]

X = df[FEATURES].fillna(0)
y = df["is_surge_trip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train XGBoost ────────────────────────────────────────────
print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42, verbosity=0
)
model.fit(X_train, y_train)
print("✅  Model trained!")

# ── SHAP Explainer ───────────────────────────────────────────
print("\nCalculating SHAP values (this takes ~30 seconds)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print("✅  SHAP values calculated!")

# ── Plot 1 — Summary Bar Plot ────────────────────────────────
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values, X_test,
    plot_type="bar",
    feature_names=FEATURES,
    show=False,
    color="#4F86C6"
)
plt.title("SHAP Feature Importance — Surge Classifier", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/shap_bar.png", bbox_inches="tight")
plt.close()
print("✅  SHAP bar chart → outputs/shap_bar.png")

# ── Plot 2 — Summary Dot Plot ────────────────────────────────
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_test,
    feature_names=FEATURES,
    show=False
)
plt.title("SHAP Summary Plot — Impact on Surge Prediction", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", bbox_inches="tight")
plt.close()
print("✅  SHAP summary plot → outputs/shap_summary.png")

# ── Plot 3 — Single Prediction Explained ─────────────────────
# Pick one surge trip and one normal trip and explain them
surge_idx  = X_test[y_test == 1].index[0]
normal_idx = X_test[y_test == 0].index[0]

surge_pos  = list(X_test.index).index(surge_idx)
normal_pos = list(X_test.index).index(normal_idx)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

shap.waterfall_plot(
    shap.Explanation(
        values        = shap_values[surge_pos],
        base_values   = explainer.expected_value,
        data          = X_test.iloc[surge_pos].values,
        feature_names = FEATURES
    ),
    show=False
)
plt.title("Why this trip was predicted as SURGE", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/shap_waterfall_surge.png", bbox_inches="tight")
plt.close()
print("✅  Surge waterfall → outputs/shap_waterfall_surge.png")

shap.waterfall_plot(
    shap.Explanation(
        values        = shap_values[normal_pos],
        base_values   = explainer.expected_value,
        data          = X_test.iloc[normal_pos].values,
        feature_names = FEATURES
    ),
    show=False
)
plt.title("Why this trip was predicted as NORMAL", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/shap_waterfall_normal.png", bbox_inches="tight")
plt.close()
print("✅  Normal waterfall → outputs/shap_waterfall_normal.png")

# ── Plot 4 — Dependence Plot (hour vs surge) ─────────────────
plt.figure(figsize=(10, 5))
shap.dependence_plot(
    "hour", shap_values, X_test,
    feature_names=FEATURES,
    interaction_index="zone_type_enc",
    show=False
)
plt.title("How Hour of Day Impacts Surge Prediction", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/shap_dependence_hour.png", bbox_inches="tight")
plt.close()
print("✅  Dependence plot → outputs/shap_dependence_hour.png")

# ── Print summary ────────────────────────────────────────────
mean_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=FEATURES
).sort_values(ascending=False)

print("\n📊 SHAP Feature Importance (mean |SHAP value|):")
for feat, val in mean_shap.items():
    bar = "█" * int(val * 50)
    print(f"  {feat:<25} {val:.4f}  {bar}")

print("\n💡 How to explain SHAP in interviews:")
print("  'I used SHAP values to explain my XGBoost model predictions.")
print("   For example, trips during hour 22-23 increased surge probability")
print("   by X%, while residential zones decreased it by Y%.'")
print("\n✅  SHAP analysis complete!")