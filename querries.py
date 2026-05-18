import pandas as pd
import sqlite3

df = pd.read_csv("data/trips_clean.csv")
conn = sqlite3.connect(":memory:")
df.to_sql("trips", conn, index=False, if_exists="replace")
print("✅ Data loaded!\n")

# Q1
q1 = pd.read_sql_query("""
SELECT zone_type,
       COUNT(*) AS total_trips,
       ROUND(SUM(final_fare_usd),2) AS total_revenue,
       ROUND(AVG(final_fare_usd),2) AS avg_fare,
       ROUND(AVG(surge_multiplier),3) AS avg_surge
FROM trips
GROUP BY zone_type
ORDER BY total_revenue DESC
""", conn)
print("Q1 - Revenue by Zone Type")
print(q1.to_string(index=False))

# Q2
q2 = pd.read_sql_query("""
SELECT hour,
       COUNT(*) AS trips,
       ROUND(AVG(surge_multiplier),3) AS avg_surge,
       ROUND(SUM(final_fare_usd),2) AS hourly_revenue
FROM trips
GROUP BY hour
ORDER BY avg_surge DESC
LIMIT 10
""", conn)
print("\nQ2 - Top 10 Peak Hours by Surge")
print(q2.to_string(index=False))

# Q3
q3 = pd.read_sql_query("""
SELECT pickup_zone, dropoff_zone,
       COUNT(*) AS trip_count,
       ROUND(AVG(final_fare_usd),2) AS avg_fare,
       ROUND(AVG(surge_multiplier),3) AS avg_surge
FROM trips
GROUP BY pickup_zone, dropoff_zone
ORDER BY trip_count DESC
LIMIT 10
""", conn)
print("\nQ3 - Top 10 Routes")
print(q3.to_string(index=False))

# Q4
q4 = pd.read_sql_query("""
SELECT CASE WHEN is_weekend=1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
       COUNT(*) AS total_trips,
       ROUND(AVG(final_fare_usd),2) AS avg_fare,
       ROUND(AVG(surge_multiplier),3) AS avg_surge,
       ROUND(AVG(tip_usd),2) AS avg_tip
FROM trips
GROUP BY is_weekend
""", conn)
print("\nQ4 - Weekend vs Weekday")
print(q4.to_string(index=False))

# Q5
q5 = pd.read_sql_query("""
SELECT pickup_zone, zone_type,
       CASE
           WHEN hour BETWEEN 6 AND 11 THEN 'Morning'
           WHEN hour BETWEEN 12 AND 17 THEN 'Afternoon'
           WHEN hour BETWEEN 18 AND 21 THEN 'Evening'
           ELSE 'Night'
       END AS time_block,
       COUNT(*) AS trips,
       ROUND(AVG(surge_multiplier),3) AS avg_surge
FROM trips
GROUP BY pickup_zone, zone_type, time_block
ORDER BY time_block, trips DESC
""", conn)
print("\nQ5 - Zone Demand by Time Block")
print(q5.to_string(index=False))

# Q6
q6 = pd.read_sql_query("""
SELECT payment_type,
       COUNT(*) AS trips,
       ROUND(AVG(final_fare_usd),2) AS avg_fare,
       ROUND(AVG(tip_usd),2) AS avg_tip
FROM trips
GROUP BY payment_type
ORDER BY trips DESC
""", conn)
print("\nQ6 - Payment Type and Tips")
print(q6.to_string(index=False))

# Q7
q7 = pd.read_sql_query("""
SELECT DATE(pickup_datetime) AS trip_date,
       COUNT(*) AS daily_trips,
       ROUND(SUM(final_fare_usd),2) AS daily_revenue
FROM trips
GROUP BY DATE(pickup_datetime)
ORDER BY trip_date
LIMIT 20
""", conn)
print("\nQ7 - Daily Revenue Trend")
print(q7.to_string(index=False))

# Q8
q8 = pd.read_sql_query("""
SELECT pickup_zone, hour,
       ROUND(AVG(surge_multiplier),3) AS avg_surge,
       COUNT(*) AS trips
FROM trips
GROUP BY pickup_zone, hour
HAVING avg_surge >= 1.8
ORDER BY avg_surge DESC
""", conn)
print("\nQ8 - High Surge Hotspots")
print(q8.to_string(index=False))

print("\n✅ All 8 queries done!")
conn.close()