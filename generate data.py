"""
Urban Mobility - Synthetic Dataset Generator
Generates realistic ride-share trip data for NYC-style city grid
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

ZONES = {
    1:  ("Downtown Core",       40.7128, -74.0060, "business"),
    2:  ("Midtown",             40.7549, -73.9840, "business"),
    3:  ("Upper East Side",     40.7736, -73.9566, "residential"),
    4:  ("Brooklyn Heights",    40.6960, -73.9937, "residential"),
    5:  ("JFK Airport",         40.6413, -73.7781, "transport"),
    6:  ("LaGuardia Airport",   40.7769, -73.8740, "transport"),
    7:  ("Williamsburg",        40.7081, -73.9571, "entertainment"),
    8:  ("Harlem",              40.8116, -73.9465, "residential"),
    9:  ("Financial District",  40.7075, -74.0021, "business"),
    10: ("Times Square",        40.7580, -73.9855, "entertainment"),
}

ZONE_IDS   = list(ZONES.keys())
ZONE_NAMES = {k: v[0] for k, v in ZONES.items()}

def surge_multiplier(hour, zone_type):
    if zone_type == "business":
        if 7 <= hour <= 9:   return round(np.random.uniform(1.8, 2.8), 2)
        if 17 <= hour <= 19: return round(np.random.uniform(2.0, 3.2), 2)
        if 12 <= hour <= 13: return round(np.random.uniform(1.2, 1.6), 2)
    if zone_type == "entertainment":
        if 22 <= hour or hour <= 2: return round(np.random.uniform(2.0, 3.5), 2)
        if 18 <= hour <= 21:        return round(np.random.uniform(1.5, 2.2), 2)
    if zone_type == "transport":
        if 6 <= hour <= 8:   return round(np.random.uniform(1.5, 2.0), 2)
        if 16 <= hour <= 19: return round(np.random.uniform(1.8, 2.6), 2)
    if 0 <= hour <= 5:       return round(np.random.uniform(1.3, 1.8), 2)
    return round(np.random.uniform(1.0, 1.2), 2)

def generate_trips(n=50000):
    records = []
    start_date = datetime(2023, 1, 1)
    for i in range(n):
        pickup_zone  = random.choice(ZONE_IDS)
        dropoff_zone = random.choice([z for z in ZONE_IDS if z != pickup_zone])
        zone_type    = ZONES[pickup_zone][3]

        ts = start_date + timedelta(
            days=random.randint(0, 364),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        hour    = ts.hour
        weekday = ts.weekday()

        surge        = surge_multiplier(hour, zone_type)
        base_fare    = round(np.random.uniform(5, 40), 2)
        distance_km  = round(np.random.uniform(1, 35), 2)
        duration_min = round(distance_km * np.random.uniform(2.5, 5.5), 1)
        final_fare   = round(base_fare * surge, 2)
        tip          = round(final_fare * np.random.uniform(0, 0.25), 2) if random.random() > 0.3 else 0

        records.append({
            "trip_id":           f"TRP{i+1:06d}",
            "pickup_datetime":   ts.strftime("%Y-%m-%d %H:%M:%S"),
            "dropoff_datetime":  (ts + timedelta(minutes=duration_min)).strftime("%Y-%m-%d %H:%M:%S"),
            "pickup_zone_id":    pickup_zone,
            "pickup_zone":       ZONE_NAMES[pickup_zone],
            "dropoff_zone_id":   dropoff_zone,
            "dropoff_zone":      ZONE_NAMES[dropoff_zone],
            "zone_type":         zone_type,
            "distance_km":       distance_km,
            "duration_min":      duration_min,
            "hour":              hour,
            "weekday":           weekday,
            "is_weekend":        int(weekday >= 5),
            "surge_multiplier":  surge,
            "base_fare_usd":     base_fare,
            "final_fare_usd":    final_fare,
            "tip_usd":           tip,
            "payment_type":      random.choice(["card", "card", "card", "cash", "wallet"]),
            "passenger_count":   random.choices([1,2,3,4], weights=[60,25,10,5])[0],
            "driver_rating":     round(np.random.uniform(3.5, 5.0), 1),
        })

    df = pd.DataFrame(records)
    df.to_csv("data/trips_raw.csv", index=False)
    print(f"✅  Generated {len(df):,} trip records → data/trips_raw.csv")
    return df

if __name__ == "__main__":
    generate_trips()