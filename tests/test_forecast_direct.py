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

response = client.get("/forecast?district=Jalna")
print("Jalna Response status:", response.status_code)
print("Jalna Response JSON:")
print(json.dumps(response.get_json(), indent=2))

response_beed = client.get("/forecast?district=Beed")
print("Beed Response status:", response_beed.status_code)
print("Beed Response JSON:")
print(json.dumps(response_beed.get_json(), indent=2))
