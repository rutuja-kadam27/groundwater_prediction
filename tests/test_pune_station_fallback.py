import os
import sys
import json
import pandas as pd

# Add parent path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app

app.app.testing = True
client = app.app.test_client()

# Mock session
with client.session_transaction() as sess:
    sess["user_id"] = 1

# Test a non-existent/empty station name in Pune to trigger fallback
district = "Pune"
empty_station = "NonExistentStationInPune"

if __name__ == "__main__":
    print(f"Testing fallback for {district} - Station: {empty_station}...")
    response = client.get(f"/forecast?district={district}&station={empty_station}")
    print("Status Code:", response.status_code)
    json_data = response.get_json()

    if response.status_code == 200:
        print("[OK] Success! Successfully fell back to district data.")
        print("Model Used:", json_data.get("model_used"))
        print("Historical records length:", len(json_data.get("historical", [])))
        print("Predictions length:", len(json_data.get("forecast", [])))
    else:
        print("[FAIL] Failed. Error response:")
        print(json.dumps(json_data, indent=2))
