"""
Urban Mobility — Correlation Heatmap
Shows which features affect surge multiplier and fare the most
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/trips_clean.csv")

# ── Select numeric columns only ──────────────────────────────
numeric_cols = [
    "hour", "weekday", "is_weekend", "is_rush_hour", "is_late_night",
    "distance_km", "duration_min", "passenger_count", "driver_rating",
    "surge_multiplier", "base_fare_usd", "final_fare_usd", "tip_usd",
    "avg_speed_kmh", "fare_per_km", "zone_demand_score", "month",
    "is_surge_trip"
]

corr_df = df[numeric_cols].corr().round(2)

# ── Plot 1 — Full Correlation Heatmap ────────────────────────
plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(
    corr_df,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    annot_kws={"size": 8},
    cbar_kws={"shrink": 0.8}
)
plt.title("Feature Correlation Heatmap — Urban Mobility", fontsize=14, pad=15)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()
print("✅  Full heatmap saved → outputs/correlation_heatmap.png")

# ── Plot 2 — Surge Correlation Bar Chart ─────────────────────
surge_corr = (
    corr_df["surge_multiplier"]
    .drop("surge_multiplier")
    .sort_values()
)

colors = ["#E84855" if x > 0 else "#2A9D8F" for x in surge_corr]

plt.figure(figsize=(10, 6))
bars = plt.barh(surge_corr.index, surge_corr.values, color=colors, edgecolor="white")
plt.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
plt.title("Feature Correlation with Surge Multiplier", fontsize=13)
plt.xlabel("Correlation Coefficient")
for bar, val in zip(bars, surge_corr.values):
    plt.text(val + (0.002 if val >= 0 else -0.002),
             bar.get_y() + bar.get_height() / 2,
             f"{val:.2f}", va="center",
             ha="left" if val >= 0 else "right", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/surge_correlation.png")
plt.close()
print("✅  Surge correlation chart → outputs/surge_correlation.png")

# ── Plot 3 — Fare Correlation Bar Chart ──────────────────────
fare_corr = (
    corr_df["final_fare_usd"]
    .drop("final_fare_usd")
    .sort_values()
)

colors2 = ["#4F86C6" if x > 0 else "#F4A261" for x in fare_corr]

plt.figure(figsize=(10, 6))
bars2 = plt.barh(fare_corr.index, fare_corr.values, color=colors2, edgecolor="white")
plt.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
plt.title("Feature Correlation with Final Fare", fontsize=13)
plt.xlabel("Correlation Coefficient")
for bar, val in zip(bars2, fare_corr.values):
    plt.text(val + (0.002 if val >= 0 else -0.002),
             bar.get_y() + bar.get_height() / 2,
             f"{val:.2f}", va="center",
             ha="left" if val >= 0 else "right", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/fare_correlation.png")
plt.close()
print("✅  Fare correlation chart → outputs/fare_correlation.png")

# ── Print top correlations ───────────────────────────────────
print("\n📊 Top features correlated with SURGE:")
print(corr_df["surge_multiplier"].drop("surge_multiplier")
      .abs().sort_values(ascending=False).head(5).to_string())

print("\n📊 Top features correlated with FARE:")
print(corr_df["final_fare_usd"].drop("final_fare_usd")
      .abs().sort_values(ascending=False).head(5).to_string())

print("\n✅  Correlation analysis complete!")