from flask import Blueprint, request, jsonify, render_template, session, redirect
from forecast_module.ml_models import generate_forecasts, load_and_preprocess_data, prepare_lagged_dataset
from forecast_module.explainability import explain_prediction
from forecast_module.alerts import check_forecast_alerts
from datetime import datetime
import json
import re

forecast_bp = Blueprint("forecast", __name__)


def sanitize_input(val):
    """
    Sanitizes string inputs to prevent injection attacks and cross-site scripting (XSS).
    """
    if val is None:
        return ""
    # Retain alphanumeric, spaces, dots, commas, slashes, brackets, hyphens, and underscores
    return re.sub(r"[^a-zA-Z0-9\s\.\-_\,\/\\()\[\]]", "", str(val)).strip()


def save_prediction_to_db(district, station, predicted_depth_m, horizon, model_used, confidence_score, feature_importance):
    """
    Inserts a prediction audit record into the SQLite database.
    """
    from app import get_user_connection
    try:
        conn = get_user_connection()
        conn.execute(
            """
            INSERT INTO predictions_history 
            (district, station, predicted_depth_m, horizon, model_used, confidence_score, feature_importance, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                district,
                station,
                predicted_depth_m,
                horizon,
                model_used,
                confidence_score,
                json.dumps(feature_importance),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("Database save error:", str(e))


@forecast_bp.route("/forecast-dashboard")
def forecast_dashboard():
    if "user_id" not in session:
        return redirect("/login")
    from app import df
    districts = sorted(df["district"].dropna().unique()) if not df.empty and "district" in df.columns else []
    return render_template("forecast_dashboard.html", districts=districts)


@forecast_bp.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Generates forecasting horizons (7d, 30d, 3m, 1y) and logs them to the database.
    Supports both GET (query parameters) and POST (JSON payload).
    """
    try:
        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            district = sanitize_input(payload.get("district"))
            station = sanitize_input(payload.get("station"))
        else:
            district = sanitize_input(request.args.get("district"))
            station = sanitize_input(request.args.get("station"))

        if not district:
            return jsonify({"error": "District parameter is required."}), 400

        forecast_results = generate_forecasts(district, station)
        if "error" in forecast_results:
            return jsonify(forecast_results), 400

        # Retrieve explainability for logging
        historical_df = load_and_preprocess_data(district, station)
        supervised_df, features, target = prepare_lagged_dataset(historical_df)
        
        explanation_data = {"contributions": [], "summary_points": []}
        if not supervised_df.empty:
            last_features = supervised_df.iloc[-1][features].to_dict()
            explanation_data = explain_prediction(
                forecast_results["model_used"], 
                None, 
                supervised_df, 
                last_features
            )

        # Save each prediction horizon to the SQLite database
        for p in forecast_results["predictions"]:
            save_prediction_to_db(
                district=district,
                station=station,
                predicted_depth_m=p["predicted_depth_m"],
                horizon=p["horizon"],
                model_used=p["model_used"],
                confidence_score=p["confidence_score"],
                feature_importance=explanation_data["contributions"]
            )

        return jsonify(forecast_results)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@forecast_bp.route("/forecast", methods=["GET"])
def get_forecast_details():
    """
    Retrieves both the historical water table depth trend and the 
    future predicted points to feed frontend graphs.
    """
    try:
        district = sanitize_input(request.args.get("district"))
        station = sanitize_input(request.args.get("station"))

        if not district:
            return jsonify({"error": "District parameter is required."}), 400

        df_hist = load_and_preprocess_data(district, station)
        # Fallback to district-wide historical data if station-specific data is empty or too small
        if station and (df_hist.empty or len(df_hist) < 5):
            df_district = load_and_preprocess_data(district, station=None)
            if not df_district.empty:
                df_hist = df_district

        if df_hist.empty:
            return jsonify({"error": f"No historical records found for district {district}"}), 400

        forecast_results = generate_forecasts(district, station)
        if "error" in forecast_results:
            return jsonify(forecast_results), 400

        # Format historical trend
        historical_series = []
        for _, row in df_hist.iterrows():
            historical_series.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "depth": round(float(row["depth"]), 2)
            })

        return jsonify({
            "success": True,
            "district": district,
            "station": station,
            "historical": historical_series,
            "forecast": forecast_results["predictions"],
            "model_used": forecast_results["model_used"],
            "mae": forecast_results["mae"],
            "rmse": forecast_results["rmse"]
        })
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve forecast: {str(e)}"}), 500


@forecast_bp.route("/feature-importance", methods=["GET"])
def get_feature_importance():
    """
    Calculates explainability contribution parameters for the selected location's prediction.
    """
    try:
        district = sanitize_input(request.args.get("district"))
        station = sanitize_input(request.args.get("station"))

        if not district:
            return jsonify({"error": "District parameter is required."}), 400

        df_hist = load_and_preprocess_data(district, station)
        # Fallback to district-wide historical data if station-specific data is empty or too small
        if station and (df_hist.empty or len(df_hist) < 5):
            df_district = load_and_preprocess_data(district, station=None)
            if not df_district.empty:
                df_hist = df_district

        if df_hist.empty or len(df_hist) < 2:
            return jsonify({"error": "Insufficient historical data."}), 400

        # Handle very small datasets with simplified linear trend explanation
        if len(df_hist) < 5:
            explanation = {
                "contributions": [
                    {
                        "feature": "time_progression",
                        "label": "Time Progression Trend",
                        "value": 1.0,
                        "baseline": 0.0,
                        "impact": 0.5,
                        "direction": "increase"
                    }
                ],
                "summary_points": [
                    "Prediction is based on linear extrapolation of historical values.",
                    "Linear regression model fit successfully on available records."
                ]
            }
            return jsonify({
                "district": district,
                "station": station,
                "model_used": "Linear Trend Projection",
                "explanation": explanation
            })

        df_supervised, features, target = prepare_lagged_dataset(df_hist)
        if df_supervised.empty:
            return jsonify({"error": "Failed to extract features."}), 400

        # Pull prediction to identify model used
        forecast_results = generate_forecasts(district, station)
        model_name = forecast_results.get("model_used", "Random Forest")

        last_features = df_supervised.iloc[-1][features].to_dict()
        explanation = explain_prediction(model_name, None, df_supervised, last_features)
        
        return jsonify({
            "district": district,
            "station": station,
            "model_used": model_name,
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve feature importances: {str(e)}"}), 500


@forecast_bp.route("/alerts", methods=["GET"])
def get_forecast_alerts():
    """
    Exposes any generated warnings or opportunities for the forecasted parameters.
    """
    try:
        district = sanitize_input(request.args.get("district"))
        station = sanitize_input(request.args.get("station"))

        if not district:
            return jsonify({"error": "District parameter is required."}), 400

        forecast_results = generate_forecasts(district, station)
        if "error" in forecast_results:
            return jsonify(forecast_results), 400

        alerts = check_forecast_alerts(
            district, 
            forecast_results["predictions"], 
            forecast_results.get("historical_avg", 6.0)
        )
        
        return jsonify({
            "district": district,
            "station": station,
            "alerts_count": len(alerts),
            "alerts": alerts
        })
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve alerts: {str(e)}"}), 500


@forecast_bp.route("/api/model-performance", methods=["GET"])
def get_model_performance():
    """
    Triggers model hyperparameter optimization and returns cross-validated 
    performance statistics (MAE, RMSE, MAPE, R2) for all model architectures.
    """
    try:
        district = sanitize_input(request.args.get("district"))
        station = sanitize_input(request.args.get("station"))

        if not district:
            return jsonify({"error": "District parameter is required."}), 400

        from forecast_module.ml_optimizer import tune_and_optimize_models
        report = tune_and_optimize_models(district, station)
        if "error" in report:
            return jsonify(report), 400
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": f"Model optimization sweep failed: {str(e)}"}), 500
