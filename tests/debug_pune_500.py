import os
import sys
import json

# Add parent path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app

app.app.testing = True
client = app.app.test_client()

# Mock session
with client.session_transaction() as sess:
    sess["user_id"] = 1

district = "Pune"
empty_station = "NonExistentStationInPune"

response = client.get(f"/forecast?district={district}&station={empty_station}")
print("Status Code:", response.status_code)
try:
    print("Response JSON:")
    print(json.dumps(response.get_json(), indent=2))
except Exception as e:
    print("Failed to parse JSON:", str(e))
    print("Response Data:", response.data)
