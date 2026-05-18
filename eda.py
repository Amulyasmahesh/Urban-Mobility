"""
Urban Mobility — Pandas EDA & Feature Engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "figure.figsize": (10, 5)})

# ── STEP 1 — Load & Inspect ──────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading Data")
print("=" * 60)

df = pd.read_csv("data/trips_raw.csv", parse_dates=["pickup_datetime", "dropoff_datetime"])
print(df.shape)
print(df.dtypes)
print(df.head(3))
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())


# ── STEP 2 — Cleaning ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Cleaning")
print("=" * 60)

before = len(df)
df.drop_duplicates(subset="trip_id", inplace=True)
print(f"Duplicates removed: {before - len(df)}")

df = df[df["distance_km"] > 0.1]
df = df[df["duration_min"] > 0.5]
df = df[df["final_fare_usd"] > 0]
df = df[df["passenger_count"].between(1, 6)]
print(f"Clean rows remaining: {len(df):,}")

df["payment_type"] = df["payment_type"].astype("category")
df["zone_type"]    = df["zone_type"].astype("category")
df["is_weekend"]   = df["is_weekend"].astype(bool)


# ── STEP 3 — EDA ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — EDA")
print("=" * 60)

trips_by_hour = df.groupby("hour")["trip_id"].count().reset_index()
trips_by_hour.columns = ["hour", "trip_count"]
print("\nPeak hours:\n", trips_by_hour.sort_values("trip_count", ascending=False).head(5))

surge_pivot = df.pivot_table(
    values="surge_multiplier",
    index="zone_type",
    columns=pd.cut(df["hour"], bins=[0,6,12,18,24], labels=["Night","Morning","Afternoon","Evening"]),
    aggfunc="mean"
).round(2)
print("\nAvg surge by zone & time-of-day:\n", surge_pivot)

revenue_by_zone = (
    df.groupby("pickup_zone")
      .agg(total_revenue=("final_fare_usd", "sum"),
           trips=("trip_id", "count"),
           avg_fare=("final_fare_usd", "mean"))
      .sort_values("total_revenue", ascending=False)
      .reset_index()
)
print("\nTop 5 zones by revenue:\n", revenue_by_zone.head())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df["surge_multiplier"].hist(bins=30, ax=axes[0], color="#4F86C6", edgecolor="white")
axes[0].set_title("Surge Multiplier Distribution")
axes[0].set_xlabel("Surge")

df.groupby("hour")["surge_multiplier"].mean().plot(ax=axes[1], marker="o", color="#E84855")
axes[1].set_title("Average Surge by Hour of Day")
axes[1].set_xlabel("Hour")
axes[1].set_ylabel("Avg Surge")
plt.tight_layout()
plt.savefig("outputs/eda_surge_plots.png")
plt.close()
print("\n✅  EDA plots saved → outputs/eda_surge_plots.png")


# ── STEP 4 — Feature Engineering ─────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Feature Engineering")
print("=" * 60)

df["month"]         = df["pickup_datetime"].dt.month
df["day_of_year"]   = df["pickup_datetime"].dt.dayofyear
df["is_rush_hour"]  = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
df["is_late_night"] = df["hour"].isin([22, 23, 0, 1, 2]).astype(int)
df["avg_speed_kmh"] = (df["distance_km"] / (df["duration_min"] / 60)).round(2)
df["fare_per_km"]   = (df["final_fare_usd"] / df["distance_km"]).round(2)
df["is_surge_trip"] = (df["surge_multiplier"] >= 1.5).astype(int)

print(f"High surge trips: {df['is_surge_trip'].sum():,} ({df['is_surge_trip'].mean()*100:.1f}%)")

df_sorted = df.sort_values("pickup_datetime")
df_sorted["zone_hour_key"] = df_sorted["pickup_zone_id"].astype(str) + "_" + df_sorted["hour"].astype(str)
zone_hour_demand = df_sorted.groupby("zone_hour_key")["trip_id"].transform("count")
df_sorted["zone_demand_score"] = zone_hour_demand
df = df_sorted.copy()


# ── STEP 5 — Export ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Export")
print("=" * 60)

df.to_csv("data/trips_clean.csv", index=False)
print(f"✅  Clean dataset saved → data/trips_clean.csv ({len(df):,} rows, {len(df.columns)} cols)")

zone_summary = (
    df.groupby(["pickup_zone_id", "pickup_zone", "zone_type"])
      .agg(
          total_trips   = ("trip_id", "count"),
          total_revenue = ("final_fare_usd", "sum"),
          avg_surge     = ("surge_multiplier", "mean"),
          avg_fare      = ("final_fare_usd", "mean"),
          surge_trips   = ("is_surge_trip", "sum"),
      )
      .reset_index()
)
zone_summary.to_csv("data/zone_summary.csv", index=False)
print(f"✅  Zone summary saved → data/zone_summary.csv")

print("\n🎉  EDA complete! Next → run notebooks/02_ml_pipeline.py")