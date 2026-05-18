import pandas as pd
import sqlite3

df = pd.read_csv("data/trips_clean.csv")
conn = sqlite3.connect(":memory:")
df.to_sql("trips", conn, index=False, if_exists="replace")
print("✅ Data loaded!\n")

def run_query(title, sql):
    print("=" * 60)
    print(title)
    print("=" * 60)
    result = pd.read_sql_query(sql, conn)
    print(result.to_string(index=False))
    print()

# Q1. Driver efficiency — revenue per minute by zone
run_query("Q1. Driver Efficiency — Revenue per Minute by Zone", """
SELECT pickup_zone,
       COUNT(*) AS trips,
       ROUND(AVG(final_fare_usd), 2) AS avg_fare,
       ROUND(AVG(duration_min), 1) AS avg_duration,
       ROUND(SUM(final_fare_usd) / SUM(duration_min), 3) AS revenue_per_min
FROM trips
GROUP BY pickup_zone
ORDER BY revenue_per_min DESC
""")

# Q2. Untapped demand zones — high surge but low trips
run_query("Q2. Untapped Demand Zones — High Surge but Low Trips", """
SELECT pickup_zone,
       zone_type,
       COUNT(*) AS trips,
       ROUND(AVG(surge_multiplier), 3) AS avg_surge,
       ROUND(AVG(final_fare_usd), 2) AS avg_fare
FROM trips
GROUP BY pickup_zone, zone_type
HAVING avg_surge >= 1.5
   AND trips < (SELECT AVG(trip_count)
                FROM (SELECT COUNT(*) AS trip_count
                      FROM trips
                      GROUP BY pickup_zone))
ORDER BY avg_surge DESC
""")

# Q3. Best hour to drive in each zone
run_query("Q3. Best Hour to Drive in Each Zone", """
WITH hour_zone AS (
    SELECT pickup_zone,
           hour,
           ROUND(AVG(final_fare_usd), 2) AS avg_fare,
           ROUND(AVG(surge_multiplier), 3) AS avg_surge,
           COUNT(*) AS trips,
           RANK() OVER (PARTITION BY pickup_zone ORDER BY AVG(final_fare_usd) DESC) AS rank
    FROM trips
    GROUP BY pickup_zone, hour
)
SELECT pickup_zone, hour, avg_fare, avg_surge, trips
FROM hour_zone
WHERE rank = 1
ORDER BY avg_fare DESC
""")

# Q4. Passenger count impact on tips
run_query("Q4. Passenger Count Impact on Tips", """
SELECT passenger_count,
       COUNT(*) AS trips,
       ROUND(AVG(tip_usd), 2) AS avg_tip,
       ROUND(AVG(final_fare_usd), 2) AS avg_fare,
       ROUND(100.0 * AVG(tip_usd) / AVG(final_fare_usd), 1) AS tip_pct
FROM trips
GROUP BY passenger_count
ORDER BY passenger_count
""")

# Q5. Month wise revenue trend
run_query("Q5. Month Wise Revenue Trend", """
SELECT month,
       COUNT(*) AS trips,
       ROUND(SUM(final_fare_usd), 2) AS total_revenue,
       ROUND(AVG(final_fare_usd), 2) AS avg_fare,
       ROUND(AVG(surge_multiplier), 3) AS avg_surge
FROM trips
GROUP BY month
ORDER BY month
""")

# Q6. Zone pairs with highest avg fare
run_query("Q6. Zone Pairs with Highest Average Fare", """
SELECT pickup_zone,
       dropoff_zone,
       COUNT(*) AS trips,
       ROUND(AVG(final_fare_usd), 2) AS avg_fare,
       ROUND(AVG(distance_km), 2) AS avg_distance
FROM trips
GROUP BY pickup_zone, dropoff_zone
HAVING trips >= 50
ORDER BY avg_fare DESC
LIMIT 10
""")

# Q7. Late night vs rush hour comparison
run_query("Q7. Late Night vs Rush Hour Comparison", """
SELECT CASE
           WHEN is_rush_hour = 1 THEN 'Rush Hour'
           WHEN is_late_night = 1 THEN 'Late Night'
           ELSE 'Normal Hours'
       END AS time_category,
       COUNT(*) AS trips,
       ROUND(AVG(surge_multiplier), 3) AS avg_surge,
       ROUND(AVG(final_fare_usd), 2) AS avg_fare,
       ROUND(AVG(tip_usd), 2) AS avg_tip
FROM trips
GROUP BY time_category
ORDER BY avg_surge DESC
""")

# Q8. Top driver rated zones
run_query("Q8. Driver Rating by Zone", """
SELECT pickup_zone,
       COUNT(*) AS trips,
       ROUND(AVG(driver_rating), 3) AS avg_rating,
       ROUND(MIN(driver_rating), 1) AS min_rating,
       ROUND(MAX(driver_rating), 1) AS max_rating
FROM trips
GROUP BY pickup_zone
ORDER BY avg_rating DESC
""")

print("✅ All extra SQL queries done!")
conn.close()