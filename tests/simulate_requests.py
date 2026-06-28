import os
import sys
import json
import pandas as pd

# Add parent path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app

app.app.testing = True
client = app.app.test_client()

# Load cleaned dataset to get all districts and stations
df = pd.read_csv("groundwater_cleaned.csv")
districts = df["district"].dropna().unique()

print(f"Testing {len(districts)} districts...")

failed = []

for district in districts:
    # Test district-only forecast
    res_forecast = client.get(f"/forecast?district={district}")
    res_explain = client.get(f"/feature-importance?district={district}")
    res_alerts = client.get(f"/alerts?district={district}")
    
    print(f"District: {district} | Forecast: {res_forecast.status_code} | Explain: {res_explain.status_code} | Alerts: {res_alerts.status_code}")
    
    if res_forecast.status_code >= 500 or res_explain.status_code >= 500 or res_alerts.status_code >= 500:
        failed.append((district, None, res_forecast.status_code, res_explain.status_code, res_alerts.status_code))
        
    # Test stations in this district
    stations = df[df["district"] == district]["stn_name"].dropna().unique()
    for station in stations[:3]: # test up to 3 stations per district
        res_st_forecast = client.get(f"/forecast?district={district}&station={station}")
        res_st_explain = client.get(f"/feature-importance?district={district}&station={station}")
        res_st_alerts = client.get(f"/alerts?district={district}&station={station}")
        
        if res_st_forecast.status_code >= 500 or res_st_explain.status_code >= 500 or res_st_alerts.status_code >= 500:
            failed.append((district, station, res_st_forecast.status_code, res_st_explain.status_code, res_st_alerts.status_code))

print("\n--- Simulation Summary ---")
if failed:
    print("Failed requests:")
    for item in failed:
        print(f"District: {item[0]}, Station: {item[1]} | Forecast: {item[2]}, Explain: {item[3]}, Alerts: {item[4]}")
else:
    print("All simulated requests completed without 500 errors!")
