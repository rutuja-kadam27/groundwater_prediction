import unittest
import os
import json
import sqlite3
import pandas as pd
import numpy as np
import sys

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app
from forecast_module.ml_optimizer import run_data_quality_audit, tune_and_optimize_models
from forecast_module.routes import sanitize_input


class TestProjectAuditAndOptimization(unittest.TestCase):

    def setUp(self):
        app.app.testing = True
        self.client = app.app.test_client()
        
        # Setup session user
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["username"] = "audit_tester"

    def test_input_sanitization(self):
        """
        Verify that sanitize_input correctly strips unsafe characters.
        """
        unsafe_val = "Akola%;' -- <script>alert(1)</script>"
        sanitized = sanitize_input(unsafe_val)
        self.assertEqual(sanitized, "Akola -- scriptalert(1)/script")
        
        # Check normal strings
        self.assertEqual(sanitize_input("Amravati-Station_1.2"), "Amravati-Station_1.2")
        self.assertEqual(sanitize_input(None), "")

    def test_security_headers(self):
        """
        Ensure security middleware adds standard security headers to HTTP responses.
        """
        response = self.client.get("/")
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.headers.get("X-Frame-Options"), "DENY")
        self.assertEqual(response.headers.get("X-XSS-Protection"), "1; mode=block")
        self.assertTrue("Content-Security-Policy" in response.headers)

    def test_data_quality_audit(self):
        """
        Check that the data quality sweeper generates a report and returns a clean DataFrame.
        """
        df_clean, report = run_data_quality_audit("groundwater_cleaned.csv")
        
        # Verify clean dataframe
        self.assertIsInstance(df_clean, pd.DataFrame)
        self.assertFalse(df_clean.empty)
        
        # Verify report parameters
        self.assertIn("timestamp", report)
        self.assertIn("missing_fields", report)
        self.assertIn("duplicates_removed", report)
        self.assertIn("outliers_found", report)
        
        # Check that report JSON file was written
        report_path = os.path.join("static", "assets", "data_quality_report.json")
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r") as f:
            data = json.load(f)
            self.assertEqual(data["cleaned_rows"], len(df_clean))

    def test_model_optimization_tuning(self):
        """
        Validate the cross-validation hyperparameter search pipeline and plots generation.
        """
        # Read a valid district from cleaned file
        df = pd.read_csv("groundwater_cleaned.csv")
        valid_districts = df["district"].dropna().unique()
        if len(valid_districts) == 0:
            self.skipTest("No districts available in groundwater_cleaned.csv to test tuning.")
            
        test_district = valid_districts[0]
        report = tune_and_optimize_models(test_district)
        
        # Verify metrics file creation
        perf_path = os.path.join("static", "assets", "model_performance.json")
        self.assertTrue(os.path.exists(perf_path))
        
        # Ensure plot files are created
        self.assertTrue(os.path.exists(os.path.join("static", "assets", "plots", "actual_vs_predicted.png")))
        self.assertTrue(os.path.exists(os.path.join("static", "assets", "plots", "residual_plot.png")))
        
        if "error" not in report:
            self.assertIn("best_model_name", report)
            self.assertIn("best_model_metrics", report)
            self.assertIn("all_models_cv", report)

    def test_model_performance_api(self):
        """
        Test the GET /api/model-performance endpoint is secure and returns metrics.
        """
        df = pd.read_csv("groundwater_cleaned.csv")
        valid_districts = df["district"].dropna().unique()
        if len(valid_districts) == 0:
            self.skipTest("No districts available to test API.")
            
        test_district = valid_districts[0]
        response = self.client.get(f"/api/model-performance?district={test_district}")
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data.decode("utf-8"))
        self.assertIn("best_model_name", data)
        self.assertIn("all_models_cv", data)


if __name__ == "__main__":
    unittest.main()
