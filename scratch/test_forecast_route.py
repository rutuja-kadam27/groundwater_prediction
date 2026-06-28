# scratch/test_forecast_route.py
import unittest
from app import app

class TestForecastRoute(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        app.config['TESTING'] = True

    def test_forecast_endpoint_success(self):
        # Bypass the login check by setting user_id in the session
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["username"] = "audit_tester"
            
        # We query for Pune and a valid station to see if the JSON response is successful
        response = self.client.get('/forecast?district=Pune&station=Pawana%20river%20at%20Chinchwadgaon,%20Village.%20Chinchwadgaon,Taluka.%20Haweli,District.%20Pune.')
        print("Status Code:", response.status_code)
        print("Content Type:", response.content_type)
        
        # Check if the status code is 200
        self.assertEqual(response.status_code, 200)
        
        # Verify it returns JSON
        self.assertTrue(response.is_json)
        
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("forecast", data)
        self.assertIn("historical", data)
        print("Test passed! Forecast returns valid JSON.")

if __name__ == '__main__':
    unittest.main()
