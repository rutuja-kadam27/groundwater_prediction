from flask import Flask, render_template, request, redirect, session, jsonify, send_file
import json
import os
import random
import re
import smtplib
import sqlite3
from datetime import datetime
from email.message import EmailMessage
from io import BytesIO
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = "supersecretkey"

@app.after_request
def add_security_headers(response):
    """
    Appends HTTP security headers to response headers to harden security.
    """
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Ensure images, scripts, style, and charts are loaded securely
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' https: 'unsafe-inline' 'unsafe-eval'; "
        "img-src 'self' data: https:; "
        "style-src 'self' https: 'unsafe-inline'; "
        "font-src 'self' https: data:;"
    )
    return response

from forecast_module.routes import forecast_bp
app.register_blueprint(forecast_bp)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
USER_DB_PATH = os.getenv("USER_DB_PATH", "users.db")
if os.path.exists("/data") and os.path.isdir("/data"):
    USER_DB_PATH = "/data/users.db"

ALERT_DEPTH_THRESHOLD_M = float(os.getenv("ALERT_DEPTH_THRESHOLD_M", "5.0"))
ALERT_COOLDOWN_MINUTES = int(os.getenv("ALERT_COOLDOWN_MINUTES", "30"))
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
LAST_ALERT_SENT_AT = {}


def get_user_connection():
    conn = sqlite3.connect(USER_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_user_db():
    conn = get_user_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            district TEXT NOT NULL,
            station TEXT,
            predicted_depth_m REAL NOT NULL,
            horizon TEXT NOT NULL,
            model_used TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            feature_importance TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()
    
    # Run optimized index injection
    from init_db_indexes import create_db_indexes
    create_db_indexes()

FEATURES = [
    "temperature",
    "conductivity",
    "turbidity",
    "total_dissolved_solids",
    "hardness_caco3",
    "chlorides",
    "sulphate",
]

TARGETS = ["ph", "dissolved_o2", "bod", "nitrate_n"]
DEPTH_TARGET = "approx_depth"
NUMERIC_COLS = TARGETS + ["approx_depth"] + FEATURES


def load_dataset():
    try:
        data = pd.read_csv("groundwater_cleaned.csv")
        if "sampling_date" in data.columns:
            data["sampling_date"] = pd.to_datetime(data["sampling_date"], errors="coerce")
        print("Dataset loaded")
        return data
    except Exception:
        print("No dataset found")
        return pd.DataFrame()


df = load_dataset()
init_user_db()
models = {}


def load_model(name, file):
    if os.path.exists(file):
        models[name] = joblib.load(file)
        print(name, "model loaded")


for target in TARGETS:
    load_model(target, f"{target}_model.pkl")
load_model(DEPTH_TARGET, f"{DEPTH_TARGET}_model.pkl")


def convert_to_decimal(coord):
    match = re.match(r"(\d+)°([\d\.]+)", str(coord))
    if match:
        degree = float(match.group(1))
        minutes = float(match.group(2))
        return degree + (minutes / 60)
    return None


def parse_depth(value):
    """
    Convert dataset depth strings (e.g., 'Less than 50cm', '50-100cm', 'NO Information')
    into a numeric depth in meters.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    # Handle already-numeric values
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)

    s = str(value).strip()
    if not s:
        return None

    s_lower = s.lower()
    if "no information" in s_lower or "no info" in s_lower:
        return None

    # Normalize commas to dots
    s_lower = s_lower.replace(",", ".")

    # Extract "X-Ycm" ranges like "50-100cm"
    if "cm" in s_lower and "-" in s_lower:
        # Important: do NOT allow a leading '-' here.
        # In strings like "50-100cm", the hyphen is a range separator, not a negative number.
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s_lower)]
        if len(nums) >= 2:
            low, high = min(nums), max(nums)
            mid_cm = (low + high) / 2.0
            return mid_cm / 100.0

    # "Less than 50cm"
    if "less than" in s_lower:
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s_lower)]
        if nums:
            # Heuristic: map to half of the upper bound (e.g., <50cm -> 25cm)
            return (nums[0] * 0.5) / 100.0

    # "Greater than 100cm"
    if "greater than" in s_lower:
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s_lower)]
        if nums:
            # Heuristic: map to 1.25x the lower bound (e.g., >100cm -> 125cm)
            return (nums[0] * 1.25) / 100.0

    # Plain "123cm"
    if "cm" in s_lower:
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s_lower)]
        if nums:
            return nums[0] / 100.0

    # Fallback: try parse any number; assume it's already in meters
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s_lower)]
    if nums:
        return nums[0]

    return None


def calculate_wqi(ph, do, bod, nitrate):
    ph_score = max(0, 100 - abs(ph - 7) * 15)
    do_score = min(100, do * 10)
    bod_score = max(0, 100 - bod * 10)
    nitrate_score = max(0, 100 - nitrate * 50)
    return round(
        0.25 * ph_score + 0.30 * do_score + 0.25 * bod_score + 0.20 * nitrate_score,
        2,
    )


def classify_wqi(wqi):
    if wqi >= 80:
        return "Excellent"
    if wqi >= 60:
        return "Good"
    if wqi >= 40:
        return "Poor"
    return "Unfit"


def water_recommendation(ph, do, bod, nitrate, depth):
    rec = []
    if ph < 6:
        rec.append("Water is acidic. Avoid drinking and use treatment.")
    if ph > 8.5:
        rec.append("Water is alkaline. Use filtration.")
    if do < 5:
        rec.append("Low dissolved oxygen detected.")
    if bod > 4:
        rec.append("High BOD detected. Organic pollution.")
    if nitrate > 45:
        rec.append("High nitrate level. Avoid drinking.")
    if depth < 5:
        rec.append("Groundwater level is low. Use rainwater harvesting.")
    if not rec:
        rec.append("Water quality within safe limits.")
    return rec


def generate_crop_advisory(weather, wqi, predicted_depth, quality_prediction=None):
    """
    Translate weather, groundwater depth, and water quality into farming guidance.
    Uses a highly accurate multi-parameter crop suitability scoring model.
    """
    temperature = float((weather or {}).get("temperature", 0.0) or 0.0)
    humidity = float((weather or {}).get("humidity", 0.0) or 0.0)
    rainfall = float((weather or {}).get("rainfall", 0.0) or 0.0)
    wqi = float(wqi or 0.0)
    predicted_depth = float(predicted_depth or 0.0)

    crops_db = {
        "Rice": {
            "temp_opt": (22, 32), "temp_limit": (16, 38),
            "water_need": "high", "salt_tolerance": "medium",
            "desc": "Requires continuous water supply; ideal for shallow water tables or high rain."
        },
        "Wheat": {
            "temp_opt": (15, 24), "temp_limit": (5, 32),
            "water_need": "medium", "salt_tolerance": "medium",
            "desc": "Thrives in cooler dry seasons with moderate, planned irrigation."
        },
        "Cotton": {
            "temp_opt": (21, 35), "temp_limit": (15, 40),
            "water_need": "medium", "salt_tolerance": "high",
            "desc": "Enjoys warm weather; moderately drought-hardy and salt-tolerant."
        },
        "Sugarcane": {
            "temp_opt": (25, 38), "temp_limit": (20, 44),
            "water_need": "high", "salt_tolerance": "medium",
            "desc": "High thermal energy requirement; needs extensive water over a long season."
        },
        "Maize": {
            "temp_opt": (18, 30), "temp_limit": (10, 36),
            "water_need": "medium", "salt_tolerance": "medium",
            "desc": "Requires well-drained soils and steady, moderate water supply."
        },
        "Soybean": {
            "temp_opt": (20, 32), "temp_limit": (12, 38),
            "water_need": "medium", "salt_tolerance": "low",
            "desc": "Sensitive to salinity; requires moderate water during pod-filling stages."
        },
        "Gram (Chickpea)": {
            "temp_opt": (15, 25), "temp_limit": (10, 30),
            "water_need": "low", "salt_tolerance": "medium",
            "desc": "Drought-hardy winter pulse; highly sensitive to waterlogging or heavy rains."
        },
        "Millets (Bajra/Jowar)": {
            "temp_opt": (24, 35), "temp_limit": (15, 42),
            "water_need": "low", "salt_tolerance": "high",
            "desc": "Extremely drought-resilient; suitable for arid lands and deep aquifers."
        }
    }

    # 1. Base irrigation and water accessibility metrics
    if predicted_depth <= 2.0:
        irrigation_potential = 1.0
        irrigation_notes = ["Groundwater is shallow, supporting easy and low-cost irrigation access."]
    elif predicted_depth <= 5.0:
        irrigation_potential = 0.8
        irrigation_notes = ["Groundwater table is at moderate depth, practical for routine field pumping."]
    elif predicted_depth <= 10.0:
        irrigation_potential = 0.5
        irrigation_notes = ["Groundwater is deep. Pumping costs are moderate; water-saving methods advised."]
    else:
        irrigation_potential = 0.2
        irrigation_notes = ["Groundwater is critically deep. High pumping costs; avoid water-intensive crops."]

    # 2. Score each crop
    crop_scores_list = []
    crop_risks = []
    
    for name, specs in crops_db.items():
        # A. Temperature Suitability
        opt_min, opt_max = specs["temp_opt"]
        lim_min, lim_max = specs["temp_limit"]
        
        if opt_min <= temperature <= opt_max:
            temp_score = 1.0
        elif temperature < lim_min or temperature > lim_max:
            temp_score = 0.0
        elif temperature < opt_min:
            temp_score = (temperature - lim_min) / (opt_min - lim_min)
        else:
            temp_score = (lim_max - temperature) / (lim_max - opt_max)
            
        # B. Water Availability Suitability
        if specs["water_need"] == "high":
            if rainfall >= 15.0:
                water_score = 1.0
            else:
                water_score = 0.3 * (rainfall / 15.0) + 0.7 * irrigation_potential
        elif specs["water_need"] == "medium":
            water_score = min(1.0, 0.4 * (rainfall / 10.0) + 0.6 * irrigation_potential)
        else:  # low
            if predicted_depth <= 1.5:
                water_score = 0.5  # Penality for waterlogging risk
            else:
                water_score = min(1.0, 0.9 + 0.1 * (1.0 - irrigation_potential))
                
        # C. Water Quality (WQI) Suitability
        if wqi >= 80:
            wqi_score = 1.0
        elif wqi >= 60:
            wqi_score = 0.85
        elif wqi >= 40:
            wqi_score = 0.75 if specs["salt_tolerance"] == "high" else 0.5
        else:
            wqi_score = 0.5 if specs["salt_tolerance"] == "high" else 0.15
            
        # D. Combine Scores (Weighted: Temp 35%, Water 35%, WQI 30%)
        final_score = int(round((0.35 * temp_score + 0.35 * water_score + 0.30 * wqi_score) * 100))
        
        # Class determination
        if final_score >= 80:
            recommendation_class = "Highly Recommended"
            badge_color = "success"
        elif final_score >= 60:
            recommendation_class = "Suitable"
            badge_color = "primary"
        elif final_score >= 45:
            recommendation_class = "Marginal / Caution"
            badge_color = "warning"
        else:
            recommendation_class = "Not Recommended"
            badge_color = "danger"
            
        crop_scores_list.append({
            "crop": name,
            "score": final_score,
            "class": recommendation_class,
            "badge_color": badge_color,
            "description": specs["desc"]
        })

    # Sort crops by score (highest first)
    crop_scores_list = sorted(crop_scores_list, key=lambda x: x["score"], reverse=True)
    
    # Get suggested crops list for backwards compatibility
    suggested_crops = [item["crop"] for item in crop_scores_list if item["score"] >= 60]
    if not suggested_crops:
        suggested_crops = [item["crop"] for item in crop_scores_list[:2]] # Fallback to top 2

    # Farm risks and actions based on environment
    if temperature >= 38:
        crop_risks.append("Extreme heat stress detected. Increases crop evaporation rates.")
    if wqi < 50:
        crop_risks.append("Poor water quality (low WQI). May stunt development of sensitive crops.")
    if predicted_depth > 6.0:
        crop_risks.append("Deep groundwater table increases pumping costs.")
        
    actions = []
    if rainfall == 0:
        actions.append("Schedule active irrigation as no live rainfall is recorded.")
    if predicted_depth > 5.0:
        actions.append("Adopt water-saving drip or sprinkler irrigation to conserve aquifer resources.")
    if wqi < 60:
        actions.append("Blend groundwater with surface water or filter prior to irrigating sensitive crops.")
    if temperature >= 35:
        actions.append("Avoid watering fields during peak daylight hours to minimize evaporation.")
    if not actions:
        actions.append("Environmental parameters are optimal. Maintain standard cropping schedules.")

    # Determine general suitability classification
    avg_top_score = sum(x["score"] for x in crop_scores_list[:3]) / 3
    if avg_top_score >= 80:
        suitability = "Highly Suitable for Farming"
    elif avg_top_score >= 60:
        suitability = "Moderately Suitable for Farming"
    else:
        suitability = "Marginal Suitability - Requires Caution"

    return {
        "suitability": suitability,
        "irrigation_score": int(avg_top_score / 10),
        "summary": " ".join(irrigation_notes) if irrigation_notes else "No farming advisory available.",
        "irrigation_notes": irrigation_notes,
        "crop_risks": crop_risks,
        "suggested_crops": suggested_crops,
        "actions": actions,
        "crop_scores": crop_scores_list
    }


def district_summary_df():
    if df.empty or "district" not in df.columns:
        return pd.DataFrame()

    summary_df = df.copy()
    for col in NUMERIC_COLS:
        if col in summary_df.columns and col != "approx_depth":
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    if "approx_depth" in summary_df.columns:
        summary_df["approx_depth"] = summary_df["approx_depth"].apply(parse_depth)

    # Depth in the dataset may be categorical (e.g., 'Less than 50cm').
    # We convert it to numeric meters and then compute a true average.
    summary = (
        summary_df.dropna(subset=["district"])
        .groupby("district", as_index=False)
        .agg(
            {
                "ph": "mean",
                "dissolved_o2": "mean",
                "bod": "mean",
                "nitrate_n": "mean",
                "approx_depth": "mean",
            }
        )
    )
    if summary.empty:
        return summary

    summary["approx_depth"] = pd.to_numeric(summary.get("approx_depth"), errors="coerce")
    global_median_depth = summary["approx_depth"].median(skipna=True)
    if pd.isna(global_median_depth):
        global_median_depth = 0.0
    summary["approx_depth"] = summary["approx_depth"].fillna(global_median_depth)

    summary["wqi"] = summary.apply(
        lambda r: calculate_wqi(r["ph"], r["dissolved_o2"], r["bod"], r["nitrate_n"]),
        axis=1,
    )
    summary["quality"] = summary["wqi"].apply(classify_wqi)
    return summary


def get_district_coordinates(district_name):
    if (
        df.empty
        or "district" not in df.columns
        or "latitude" not in df.columns
        or "longitude" not in df.columns
    ):
        return None, None

    district_df = df[df["district"].astype(str).str.lower() == str(district_name).lower()].copy()
    if district_df.empty:
        return None, None

    district_df["lat_decimal"] = district_df["latitude"].apply(convert_to_decimal)
    district_df["lng_decimal"] = district_df["longitude"].apply(convert_to_decimal)
    district_df = district_df.dropna(subset=["lat_decimal", "lng_decimal"])
    if district_df.empty:
        return None, None

    return float(district_df["lat_decimal"].mean()), float(district_df["lng_decimal"].mean())


def fetch_live_weather(lat, lng):
    params = urlencode(
        {
            "latitude": lat,
            "longitude": lng,
            "current": "temperature_2m,relative_humidity_2m,rain,precipitation",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    try:
        with urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except URLError as e:
        raise RuntimeError(f"Weather API unreachable: {e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"Could not parse weather response: {str(e)}") from e

    current = payload.get("current", {})
    rainfall_value = current.get("rain", None)
    if rainfall_value is None:
        rainfall_value = current.get("precipitation", 0.0)

    rainfall = float(rainfall_value or 0.0)
    return {
        "temperature": float(current.get("temperature_2m", 0.0) or 0.0),
        "humidity": float(current.get("relative_humidity_2m", 0.0) or 0.0),
        "rainfall": rainfall,
        "rainfall_text": f"{rainfall:.2f} mm" + (" (No rain now)" if rainfall == 0 else ""),
    }


def build_realtime_features(district_name, weather):
    base = {}
    if not df.empty and "district" in df.columns:
        district_rows = df[df["district"].astype(str).str.lower() == str(district_name).lower()].copy()
        if not district_rows.empty:
            for col in FEATURES:
                if col in district_rows.columns:
                    base[col] = pd.to_numeric(district_rows[col], errors="coerce").mean()

    if not base:
        for col in FEATURES:
            if col in df.columns:
                base[col] = pd.to_numeric(df[col], errors="coerce").mean()

    # Simple mapping from weather -> model feature space.
    # This keeps the model inputs consistent and allows live update effects.
    rainfall = weather["rainfall"]
    humidity = weather["humidity"]
    temperature = weather["temperature"]

    base["temperature"] = temperature
    base["conductivity"] = max(0.0, float(base.get("conductivity", 0.0) or 0.0) + (0.05 * humidity) + (0.4 * rainfall))
    base["turbidity"] = max(0.0, float(base.get("turbidity", 0.0) or 0.0) + (0.3 * rainfall))
    base["total_dissolved_solids"] = max(
        0.0,
        float(base.get("total_dissolved_solids", 0.0) or 0.0) + (0.2 * humidity) + (0.5 * rainfall),
    )
    base["hardness_caco3"] = max(0.0, float(base.get("hardness_caco3", 0.0) or 0.0) + (0.1 * humidity))
    base["chlorides"] = max(0.0, float(base.get("chlorides", 0.0) or 0.0) + (0.08 * rainfall))
    base["sulphate"] = max(0.0, float(base.get("sulphate", 0.0) or 0.0) + (0.06 * rainfall))

    for col in FEATURES:
        value = base.get(col, 0.0)
        if pd.isna(value):
            base[col] = 0.0
        else:
            base[col] = float(value)
    return base


def _depth_training_data():
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    if DEPTH_TARGET not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=float)
    if not all(col in df.columns for col in FEATURES):
        return pd.DataFrame(), pd.Series(dtype=float)

    data = df.copy()
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Convert categorical depth into numeric meters for training.
    data[DEPTH_TARGET] = data[DEPTH_TARGET].apply(parse_depth)
    data = data.dropna(subset=FEATURES + [DEPTH_TARGET])
    if data.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    return data[FEATURES], data[DEPTH_TARGET]


def explain_depth_prediction(feature_dict, predicted_depth):
    model = models.get(DEPTH_TARGET)
    X_train, y_train = _depth_training_data()
    if model is None or not hasattr(model, "feature_importances_") or X_train.empty:
        heuristic_importance = {
            "temperature": 0.22,
            "turbidity": 0.16,
            "conductivity": 0.15,
            "total_dissolved_solids": 0.14,
            "hardness_caco3": 0.12,
            "chlorides": 0.11,
            "sulphate": 0.10,
        }
        importance_sorted = sorted(
            [{"feature": k, "importance": v} for k, v in heuristic_importance.items()],
            key=lambda x: x["importance"],
            reverse=True,
        )
        lead = ", ".join([x["feature"].replace("_", " ") for x in importance_sorted[:2]])
        return {
            "text": f"Model-specific explanation is not available yet, so this fallback explanation uses domain rules. Major influence likely comes from {lead}.",
            "importance": importance_sorted,
            "top_positive": [],
            "top_negative": [],
        }

    importances = {f: float(v) for f, v in zip(FEATURES, model.feature_importances_)}
    baseline = X_train.median(numeric_only=True).to_dict()

    contributions = []
    for feature in FEATURES:
        corr = X_train[feature].corr(y_train)
        corr_sign = 1.0 if pd.isna(corr) or corr >= 0 else -1.0
        diff = float(feature_dict.get(feature, 0.0)) - float(baseline.get(feature, 0.0))
        effect = diff * corr_sign * importances.get(feature, 0.0)
        contributions.append(
            {
                "feature": feature,
                "importance": round(importances.get(feature, 0.0), 4),
                "effect_score": round(effect, 4),
                "effect": "increase" if effect >= 0 else "decrease",
            }
        )

    contributions_sorted = sorted(contributions, key=lambda x: abs(x["effect_score"]), reverse=True)
    top_positive = [c for c in contributions_sorted if c["effect_score"] > 0][:2]
    top_negative = [c for c in contributions_sorted if c["effect_score"] < 0][:2]

    inc_words = [c["feature"].replace("_", " ") for c in top_positive]
    dec_words = [c["feature"].replace("_", " ") for c in top_negative]

    if predicted_depth > 5:
        if inc_words:
            text = "Low groundwater level (deeper depth) is mainly driven by " + ", ".join(inc_words) + "."
        else:
            text = "Low groundwater level is predicted from the current feature pattern."
    elif predicted_depth <= 2:
        if dec_words:
            text = "High groundwater level (shallower depth) is mainly supported by " + ", ".join(dec_words) + "."
        else:
            text = "High groundwater level is predicted from the current feature pattern."
    else:
        text = "Moderate groundwater level is predicted with balanced feature influence."

    importance_sorted = sorted(
        [{"feature": k, "importance": round(v, 4)} for k, v in importances.items()],
        key=lambda x: x["importance"],
        reverse=True,
    )

    return {
        "text": text,
        "importance": importance_sorted,
        "top_positive": top_positive,
        "top_negative": top_negative,
    }


def run_prediction_from_features(feature_dict, approx_depth=6.0):
    input_df = pd.DataFrame([[feature_dict[f] for f in FEATURES]], columns=FEATURES)
    ph = models["ph"].predict(input_df)[0]
    do = models["dissolved_o2"].predict(input_df)[0]
    bod = models["bod"].predict(input_df)[0]
    nitrate = models["nitrate_n"].predict(input_df)[0]
    wqi = calculate_wqi(ph, do, bod, nitrate)
    category = classify_wqi(wqi)
    return {
        "prediction": {
            "ph": round(float(ph), 2),
            "dissolved_o2": round(float(do), 2),
            "bod": round(float(bod), 2),
            "nitrate_n": round(float(nitrate), 2),
        },
        "wqi": round(float(wqi), 2),
        "category": category,
        "recommendations": water_recommendation(ph, do, bod, nitrate, approx_depth),
    }


def predict_groundwater_level(district_name, weather):
    feature_dict = build_realtime_features(district_name or "", weather)
    model = models.get(DEPTH_TARGET)

    if model is not None:
        input_df = pd.DataFrame([[feature_dict[f] for f in FEATURES]], columns=FEATURES)
        adjusted_depth = float(model.predict(input_df)[0])
        adjusted_depth = max(0.3, round(adjusted_depth, 2))
    else:
        # Fallback if depth model is unavailable.
        base_depth = 6.0
        if not df.empty and "approx_depth" in df.columns:
            if district_name and "district" in df.columns:
                district_rows = df[df["district"].astype(str).str.lower() == str(district_name).lower()]
                if not district_rows.empty:
                    depths = district_rows["approx_depth"].apply(parse_depth).dropna()
                    if not depths.empty:
                        base_depth = float(depths.mean())
            else:
                depths = df["approx_depth"].apply(parse_depth).dropna()
                if not depths.empty:
                    base_depth = float(depths.mean())
        if pd.isna(base_depth):
            base_depth = 6.0
        adjusted_depth = float(base_depth)

    if adjusted_depth <= 2:
        level_status = "High groundwater level"
        note = "Water table is relatively close to ground surface."
    elif adjusted_depth <= 5:
        level_status = "Moderate groundwater level"
        note = "Groundwater condition is moderate."
    else:
        level_status = "Low groundwater level"
        note = "Groundwater table is deeper; conservation is recommended."

    return {
        "predicted_depth_m": adjusted_depth,
        "level_status": level_status,
        "insight": note,
        "features_used": feature_dict,
    }


def get_location_depth_data(district_name, village_name=None):
    if df.empty or "approx_depth" not in df.columns:
        return []

    filtered = df.copy()
    if district_name and "district" in filtered.columns:
        filtered = filtered[filtered["district"].astype(str).str.lower() == str(district_name).lower()]
    if village_name and "stn_name" in filtered.columns:
        filtered = filtered[filtered["stn_name"].astype(str).str.lower() == str(village_name).lower()]

    if filtered.empty:
        return []

    if "sampling_date" in filtered.columns:
        filtered["sampling_date"] = pd.to_datetime(filtered["sampling_date"], errors="coerce")
        filtered = filtered.dropna(subset=["sampling_date"])
        filtered["approx_depth"] = filtered["approx_depth"].apply(parse_depth)
        filtered = filtered.dropna(subset=["approx_depth"]).sort_values("sampling_date")
        return [
            {"date": d.strftime("%Y-%m-%d"), "depth": float(v)}
            for d, v in zip(filtered["sampling_date"], filtered["approx_depth"])
        ]

    filtered["approx_depth"] = filtered["approx_depth"].apply(parse_depth)
    filtered = filtered.dropna(subset=["approx_depth"])
    return [{"date": "Unknown", "depth": float(v)} for v in filtered["approx_depth"].tolist()]


def build_trend_chart_image(trend_rows):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    if trend_rows:
        x = [row["date"] for row in trend_rows]
        y = [row["depth"] for row in trend_rows]
        ax.plot(x, y, marker="o", color="#0077b6", linewidth=2.8, markersize=5.5)
        ax.set_title("Groundwater Depth Trend", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Depth (m)", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        if len(x) > 8:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")
    else:
        ax.text(0.5, 0.5, "No trend data available", ha="center", va="center", fontsize=13)
        ax.set_axis_off()
    fig.tight_layout()
    image = BytesIO()
    fig.savefig(image, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    image.seek(0)
    return image


def build_comparison_chart_image(past_avg_depth, predicted_depth):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    labels = ["Past Average", "Predicted (Live)"]
    values = [past_avg_depth, predicted_depth]
    colors_list = ["#8ecae6", "#219ebc"]
    ax.bar(labels, values, color=colors_list)
    ax.set_title("Past vs Predicted Depth", fontsize=14, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    for i, value in enumerate(values):
        ax.text(i, value + 0.05, f"{value:.2f}", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    image = BytesIO()
    fig.savefig(image, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    image.seek(0)
    return image


def generate_groundwater_pdf_report(district, village, weather, level_prediction, trend_rows, quality_result=None, crop_advisory=None):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4, pageCompression=0)
    page_width, page_height = A4

    # Top Banner
    pdf.setFillColor(colors.HexColor("#0f172a")) # Slate 900
    pdf.rect(0, page_height - 75, page_width, 75, fill=1, stroke=0)
    
    # Title
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(40, page_height - 45, "Groundwater Resource Evaluation Report")
    
    # Subtitle
    pdf.setFont("Helvetica-Oblique", 10)
    pdf.drawString(40, page_height - 60, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Separator line
    pdf.setStrokeColor(colors.HexColor("#00f2fe"))
    pdf.setLineWidth(3)
    pdf.line(0, page_height - 75, page_width, page_height - 75)

    # Column 1 (Left): Location & Weather (X = 40)
    pdf.setFillColor(colors.HexColor("#1e293b")) # Dark Slate
    y = page_height - 110
    
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.drawString(40, y, "LOCATION & METEOROLOGICAL DETAILS")
    
    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(colors.HexColor("#334155"))
    y -= 20
    pdf.drawString(45, y, f"District: {district or 'N/A'}")
    y -= 15
    pdf.drawString(45, y, f"Station: {village or 'All Stations Average'}")
    
    y -= 25
    pdf.setFont("Helvetica-Bold", 11)
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.drawString(40, y, "Live Weather Input")
    
    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(colors.HexColor("#334155"))
    y -= 18
    pdf.drawString(45, y, f"Temperature: {weather.get('temperature', 0.0):.2f} \u00b0C")
    y -= 15
    pdf.drawString(45, y, f"Humidity: {weather.get('humidity', 0.0):.2f} %")
    y -= 15
    pdf.drawString(45, y, f"Rainfall: {weather.get('rainfall', 0.0):.2f} mm")

    # Column 2 (Right): Prediction & WQI (X = 310)
    y_right = page_height - 110
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.drawString(310, y_right, "ANALYSIS & PREDICTION RESULTS")
    
    y_right -= 20
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(310, y_right, f"Water Level Status: {level_prediction.get('level_status', 'N/A')}")
    
    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(colors.HexColor("#334155"))
    y_right -= 18
    pdf.drawString(315, y_right, f"Predicted Depth: {level_prediction.get('predicted_depth_m', 0.0):.2f} m (below ground)")
    
    if quality_result:
        y_right -= 22
        pdf.setFont("Helvetica-Bold", 11)
        pdf.setFillColor(colors.HexColor("#0f172a"))
        pdf.drawString(310, y_right, "Water Quality Prediction")
        
        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(colors.HexColor("#334155"))
        y_right -= 18
        pdf.drawString(315, y_right, f"Predicted WQI: {quality_result.get('wqi', 0.0):.2f} ({quality_result.get('category', 'N/A')})")
        
    if crop_advisory:
        y_right -= 22
        pdf.setFont("Helvetica-Bold", 11)
        pdf.setFillColor(colors.HexColor("#0f172a"))
        pdf.drawString(310, y_right, "Farming Suitability")
        
        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(colors.HexColor("#334155"))
        y_right -= 18
        pdf.drawString(315, y_right, f"Suitability: {crop_advisory.get('suitability', 'N/A')}")

    # Middle Row: Recommended Actions & Insights (y_min of both columns)
    y_mid = min(y, y_right) - 25
    pdf.setStrokeColor(colors.HexColor("#cbd5e1"))
    pdf.setLineWidth(0.5)
    pdf.line(40, y_mid + 10, page_width - 40, y_mid + 10)

    pdf.setFont("Helvetica-Bold", 11)
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.drawString(40, y_mid, "AI Insight & Recommendations")
    
    pdf.setFont("Helvetica-Oblique", 9.5)
    pdf.setFillColor(colors.HexColor("#475569"))
    y_mid -= 18
    pdf.drawString(45, y_mid, f"Insight: {level_prediction.get('insight', 'N/A')}")
    
    if crop_advisory and crop_advisory.get("actions"):
        y_mid -= 16
        actions_str = ", ".join(crop_advisory["actions"])
        # Wrap long actions text manually to avoid off-page rendering
        if len(actions_str) > 110:
            actions_str1 = actions_str[:110]
            actions_str2 = actions_str[110:]
            pdf.drawString(45, y_mid, f"Actions: {actions_str1}")
            y_mid -= 13
            pdf.drawString(85, y_mid, actions_str2)
        else:
            pdf.drawString(45, y_mid, f"Actions: {actions_str}")

    # Divider before charts
    y_charts_label = y_mid - 25
    pdf.line(40, y_charts_label + 12, page_width - 40, y_charts_label + 12)

    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.drawString(40, y_charts_label, "VISUAL DATA CHARTS")

    # Generate charts
    trend_chart = build_trend_chart_image(trend_rows)
    past_avg = (
        sum(row["depth"] for row in trend_rows) / len(trend_rows)
        if trend_rows
        else float(level_prediction["predicted_depth_m"])
    )
    compare_chart = build_comparison_chart_image(float(past_avg), float(level_prediction["predicted_depth_m"]))

    y_charts_draw = y_charts_label - 185
    # Render charts side-by-side
    pdf.drawImage(
        ImageReader(trend_chart),
        40,
        y_charts_draw,
        width=250,
        height=170,
        preserveAspectRatio=True,
        mask="auto",
    )
    pdf.drawImage(
        ImageReader(compare_chart),
        305,
        y_charts_draw,
        width=250,
        height=170,
        preserveAspectRatio=True,
        mask="auto",
    )

    # Footer
    pdf.setStrokeColor(colors.HexColor("#cbd5e1"))
    pdf.setLineWidth(0.5)
    pdf.line(40, 55, page_width - 40, 55)
    
    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(colors.HexColor("#94a3b8"))
    pdf.drawCentredString(page_width / 2.0, 40, "💧 AI Powered Groundwater Prediction using Digital Water Level Recorder (DWLR) Data")
    pdf.drawCentredString(page_width / 2.0, 28, "© 2026 Groundwater Resource Evaluation. All Rights Reserved.")

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer


@app.before_request
def require_login_for_user_pages():
    public_endpoints = {
        "login",
        "signup",
        "admin_login",
        "static",
    }
    if request.endpoint in public_endpoints:
        return None

    if request.path.startswith("/admin"):
        return None

    if "user_id" not in session:
        return redirect("/login")
    return None


@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    success = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not email or not password:
            error = "All fields are required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        else:
            conn = get_user_connection()
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ? OR email = ?",
                (username, email),
            ).fetchone()
            if existing:
                error = "Username or email already exists."
            else:
                conn.execute(
                    "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (
                        username,
                        email,
                        generate_password_hash(password),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
                conn.commit()
                success = "Account created successfully. Please login."
            conn.close()

    return render_template("signup.html", error=error, success=success)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_user_connection()
        user = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect("/")
        error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    session.pop("chat_last_district", None)
    return redirect("/login")


def _location_key_for_alert(district, lat, lng):
    if district:
        return f"district:{str(district).strip().lower()}"
    return f"coord:{round(float(lat), 3)}:{round(float(lng), 3)}"


def send_alert_email(location_label, predicted_depth_m, warning_message):
    if not ALERT_EMAIL_TO or not SMTP_USER or not SMTP_PASSWORD:
        return {"sent": False, "error": "Email settings are not configured."}

    msg = EmailMessage()
    msg["Subject"] = f"Groundwater Alert - {location_label}"
    msg["From"] = ALERT_EMAIL_FROM or SMTP_USER
    msg["To"] = ALERT_EMAIL_TO
    msg.set_content(
        "\n".join(
            [
                "Groundwater Alert Notification",
                "",
                f"Location: {location_label}",
                f"Predicted Groundwater Depth: {predicted_depth_m:.2f} m (below ground)",
                f"Alert: {warning_message}",
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ]
        )
    )

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        return {"sent": True, "error": None}
    except Exception as e:
        return {"sent": False, "error": f"Email send failed: {str(e)}"}


def check_threshold(predicted_depth_m, district=None, lat=None, lng=None):
    location_label = district or f"({round(float(lat), 4)}, {round(float(lng), 4)})"
    triggered = float(predicted_depth_m) > ALERT_DEPTH_THRESHOLD_M
    if not triggered:
        return {
            "triggered": False,
            "threshold_m": ALERT_DEPTH_THRESHOLD_M,
            "message": "Groundwater level is within threshold limits.",
            "email": {"sent": False, "error": None},
        }

    warning_message = (
        f"Warning: Water level critically low in {location_label}. "
        f"Predicted depth is {float(predicted_depth_m):.2f} m (threshold: {ALERT_DEPTH_THRESHOLD_M:.2f} m)."
    )

    location_key = _location_key_for_alert(district, lat, lng)
    now = datetime.now()
    last_sent = LAST_ALERT_SENT_AT.get(location_key)
    cooldown_active = (
        last_sent is not None
        and (now - last_sent).total_seconds() < ALERT_COOLDOWN_MINUTES * 60
    )

    if cooldown_active:
        email_status = {
            "sent": False,
            "error": f"Cooldown active. Next email after {ALERT_COOLDOWN_MINUTES} minutes.",
        }
    else:
        email_status = send_alert_email(location_label, float(predicted_depth_m), warning_message)
        if email_status.get("sent"):
            LAST_ALERT_SENT_AT[location_key] = now

    return {
        "triggered": True,
        "threshold_m": ALERT_DEPTH_THRESHOLD_M,
        "message": warning_message,
        "email": email_status,
    }


def extract_intent_and_location(query_text):
    text = str(query_text or "").strip().lower()
    if not text:
        return {"intent": "unknown", "location": None}

    weather_keywords = [
        "weather",
        "temperature",
        "humidity",
        "rain",
        "rainfall",
        "climate",
    ]
    explain_keywords = [
        "why",
        "explain",
        "reason",
        "how this prediction",
        "how this works",
    ]
    quality_keywords = [
        "water quality",
        "wqi",
        "quality",
        "safe",
        "drink",
        "drinking water",
    ]
    farming_keywords = [
        "farming",
        "crop",
        "irrigation",
        "agriculture",
        "farm",
        "suitable for crops",
    ]
    prediction_keywords = [
        "groundwater",
        "water level",
        "prediction",
        "show",
        "level",
        "depth",
    ]

    if "help" in text or "what can you do" in text or "commands" in text:
        intent = "help"
    elif any(k in text for k in explain_keywords):
        intent = "explain_prediction"
    elif any(k in text for k in farming_keywords):
        intent = "get_crop_advisory"
    elif any(k in text for k in weather_keywords):
        intent = "get_weather"
    elif any(k in text for k in quality_keywords):
        intent = "get_water_quality"
    elif any(k in text for k in prediction_keywords):
        intent = "get_prediction"
    else:
        intent = "unknown"

    location = None
    if not df.empty and "district" in df.columns:
        district_names = [str(d).strip() for d in df["district"].dropna().unique().tolist()]
        for district in district_names:
            if district.lower() in text:
                location = district
                break

    if location is None and " in " in text:
        # Fallback: text after "in"
        location = text.split(" in ", 1)[1].strip().title()

    return {"intent": intent, "location": location}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard", methods=["GET", "POST"])
def index():
    prediction = None
    wqi = None
    category = None
    recommendations = None
    error = None

    dwlr_data = []
    districts = sorted(df["district"].dropna().unique()) if not df.empty and "district" in df.columns else []
    if not df.empty and {"district", "sampling_date", "approx_depth"}.issubset(df.columns):
        dwlr = df[["district", "sampling_date", "approx_depth"]].dropna()
        dwlr = dwlr.sort_values("sampling_date")
        for _, row in dwlr.iterrows():
            depth_val = parse_depth(row["approx_depth"])
            if pd.notna(depth_val):
                dwlr_data.append({
                    "district": str(row["district"]),
                    "date": row["sampling_date"].strftime("%Y-%m-%d"),
                    "depth": round(float(depth_val), 2)
                })

    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in FEATURES]
            depth_input = float(request.form.get("approx_depth", 6))

            input_df = pd.DataFrame([values], columns=FEATURES)
            ph = models["ph"].predict(input_df)[0]
            do = models["dissolved_o2"].predict(input_df)[0]
            bod = models["bod"].predict(input_df)[0]
            nitrate = models["nitrate_n"].predict(input_df)[0]

            prediction = {
                "ph": round(float(ph), 2),
                "dissolved_o2": round(float(do), 2),
                "bod": round(float(bod), 2),
                "nitrate_n": round(float(nitrate), 2),
            }
            wqi = calculate_wqi(ph, do, bod, nitrate)
            category = classify_wqi(wqi)
            recommendations = water_recommendation(ph, do, bod, nitrate, depth_input)
        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        wqi=wqi,
        category=category,
        recommendations=recommendations,
        dwlr_data=dwlr_data,
        districts=districts,
        error=error,
    )


@app.route("/realtime-groundwater")
def realtime_groundwater():
    districts = sorted(df["district"].dropna().unique()) if not df.empty and "district" in df.columns else []
    return render_template("realtime_groundwater.html", districts=districts)


@app.route("/api/stations")
def get_stations():
    district = request.args.get("district")
    if not district:
        return jsonify([])
    if df.empty or "district" not in df.columns or "stn_name" not in df.columns:
        return jsonify([])
    stations = sorted(
        df[df["district"].astype(str).str.lower() == str(district).lower()]["stn_name"]
        .dropna()
        .unique()
        .tolist()
    )
    return jsonify(stations)


@app.route("/api/live-weather")
def live_weather():
    district = request.args.get("district")
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)

    if district and (lat is None or lng is None):
        lat, lng = get_district_coordinates(district)

    if lat is None or lng is None:
        return jsonify({"error": "Valid district or coordinates are required."}), 400

    try:
        weather = fetch_live_weather(lat, lng)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    return jsonify(
        {
            "district": district,
            "lat": round(lat, 5),
            "lng": round(lng, 5),
            "weather": weather,
        }
    )


@app.route("/api/live-predict")
def live_predict():
    if any(name not in models for name in TARGETS):
        return jsonify({"error": "Models are not loaded. Train or upload dataset first."}), 500

    district = request.args.get("district")
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    approx_depth = request.args.get("approx_depth", type=float, default=6.0)

    if district and (lat is None or lng is None):
        lat, lng = get_district_coordinates(district)

    if lat is None or lng is None:
        return jsonify({"error": "Valid district or coordinates are required for prediction."}), 400

    try:
        weather = fetch_live_weather(lat, lng)
        features = build_realtime_features(district or "", weather)
        result = run_prediction_from_features(features, approx_depth=approx_depth)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify(
        {
            "district": district,
            "lat": round(lat, 5),
            "lng": round(lng, 5),
            "weather": weather,
            "features_used": features,
            **result,
        }
    )


@app.route("/api/live-groundwater-predict")
def live_groundwater_predict():
    district = request.args.get("district")
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)

    if district and (lat is None or lng is None):
        lat, lng = get_district_coordinates(district)

    if lat is None or lng is None:
        return jsonify({"error": "Valid district or coordinates are required for groundwater prediction."}), 400

    try:
        weather = fetch_live_weather(lat, lng)
        level_prediction = predict_groundwater_level(district, weather)
        quality_result = None
        crop_advisory = None
        if all(name in models for name in TARGETS):
            quality_result = run_prediction_from_features(
                level_prediction.get("features_used", {}),
                approx_depth=level_prediction.get("predicted_depth_m", 6.0),
            )
            crop_advisory = generate_crop_advisory(
                weather,
                quality_result.get("wqi"),
                level_prediction.get("predicted_depth_m", 0.0),
                quality_prediction=quality_result,
            )
        explanation = explain_depth_prediction(
            level_prediction.get("features_used", {}),
            level_prediction.get("predicted_depth_m", 0.0),
        )
        alert_data = check_threshold(
            level_prediction.get("predicted_depth_m", 0.0),
            district=district,
            lat=lat,
            lng=lng,
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Groundwater prediction failed: {str(e)}"}), 500

    return jsonify(
        {
            "district": district,
            "lat": round(lat, 5),
            "lng": round(lng, 5),
            "weather": weather,
            **level_prediction,
            "water_quality": quality_result,
            "crop_advisory": crop_advisory,
            "explanation": explanation,
            "alert": alert_data,
        }
    )


@app.route("/explain-prediction")
def explain_prediction():
    district = request.args.get("district")
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)

    if district and (lat is None or lng is None):
        lat, lng = get_district_coordinates(district)

    if lat is None or lng is None:
        return jsonify({"error": "Valid district or coordinates are required."}), 400

    try:
        weather = fetch_live_weather(lat, lng)
        level_prediction = predict_groundwater_level(district, weather)
        explanation = explain_depth_prediction(
            level_prediction.get("features_used", {}),
            level_prediction.get("predicted_depth_m", 0.0),
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Could not explain prediction: {str(e)}"}), 500

    return jsonify(
        {
            "district": district,
            "predicted_depth_m": level_prediction.get("predicted_depth_m"),
            "level_status": level_prediction.get("level_status"),
            "features_used": level_prediction.get("features_used", {}),
            "explanation": explanation,
        }
    )


@app.route("/voice-query", methods=["POST"])
def voice_query():
    payload = request.get_json(silent=True) or {}
    query = payload.get("query", "")
    parsed = extract_intent_and_location(query)
    intent = parsed["intent"]

    if intent == "help":
        return jsonify(
            {
                "recognized_text": query,
                "intent": intent,
                "location": parsed["location"],
                "reply_text": (
                    "You can ask things like: what is the weather in Akola, "
                    "show groundwater level in Pune, explain groundwater prediction in Nashik, "
                    "tell water quality in Satara, or ask if water is suitable for farming in Akola."
                ),
                "result": None,
            }
        )

    if intent == "unknown":
        return jsonify(
            {
                "recognized_text": query,
                "intent": intent,
                "location": parsed["location"],
                "reply_text": (
                    "I understood your voice, but please ask about weather, groundwater depth, "
                    "prediction explanation, water quality, or farming suitability. Example: is this water good for farming in Akola."
                ),
                "result": None,
            }
        )

    district = parsed["location"]
    if not district:
        return jsonify(
            {
                "recognized_text": query,
                "intent": intent,
                "location": None,
                "reply_text": (
                    "Please mention a district name. Example: what is the weather in Pune, "
                    "or show groundwater level in Pune."
                ),
                "result": None,
            }
        )

    lat, lng = get_district_coordinates(district)
    if lat is None or lng is None:
        return jsonify(
            {
                "recognized_text": query,
                "intent": intent,
                "location": district,
                "reply_text": f"I could not find coordinates for {district}. Please try another district.",
                "result": None,
            }
        )

    try:
        weather = fetch_live_weather(lat, lng)
        level_prediction = predict_groundwater_level(district, weather)
        explanation = explain_depth_prediction(
            level_prediction.get("features_used", {}),
            level_prediction.get("predicted_depth_m", 0.0),
        )
        features = build_realtime_features(district or "", weather)
        quality_result = None
        if all(name in models for name in TARGETS):
            quality_result = run_prediction_from_features(
                features,
                approx_depth=level_prediction.get("predicted_depth_m", 6.0),
            )
        crop_advisory = generate_crop_advisory(
            weather,
            (quality_result or {}).get("wqi", 0.0),
            level_prediction.get("predicted_depth_m", 0.0),
            quality_prediction=quality_result,
        )

        if intent == "get_weather":
            reply_text = (
                f"Current weather in {district}: temperature {weather['temperature']:.2f} degree Celsius, "
                f"humidity {weather['humidity']:.2f} percent, and rainfall {weather['rainfall']:.2f} millimeter."
            )
        elif intent == "get_water_quality":
            if quality_result is None:
                reply_text = (
                    f"I found live weather for {district}, but water quality prediction models are not loaded yet."
                )
            else:
                reply_text = (
                    f"Predicted water quality in {district} has WQI {quality_result['wqi']:.2f}, "
                    f"category {quality_result['category']}, pH {quality_result['prediction']['ph']:.2f}, "
                    f"and dissolved oxygen {quality_result['prediction']['dissolved_o2']:.2f}. "
                    f"For farming, it is {crop_advisory['suitability'].lower()}."
                )
        elif intent == "get_crop_advisory":
            reply_text = (
                f"For farming in {district}, the result is {crop_advisory['suitability'].lower()}. "
                f"{crop_advisory['summary']} Recommended focus: {', '.join(crop_advisory['suggested_crops'])}."
            )
        elif intent == "explain_prediction":
            reply_text = (
                f"In {district}, predicted groundwater depth is {level_prediction['predicted_depth_m']:.2f} meter below ground. "
                f"{explanation.get('text', 'No explanation available.')}"
            )
        else:
            reply_text = (
                f"Groundwater level in {district} is predicted at "
                f"{level_prediction['predicted_depth_m']:.2f} meter below ground. "
                f"Status is {level_prediction['level_status']}. "
                f"For farming, it is {crop_advisory['suitability'].lower()}."
            )
    except RuntimeError as e:
        return jsonify(
            {
                "recognized_text": query,
                "intent": intent,
                "location": district,
                "reply_text": f"Weather service error: {str(e)}",
                "result": None,
            }
        ), 502
    except Exception as e:
        return jsonify(
            {
                "recognized_text": query,
                "intent": intent,
                "location": district,
                "reply_text": f"Prediction failed: {str(e)}",
                "result": None,
            }
        ), 500

    return jsonify(
        {
            "recognized_text": query,
            "intent": intent,
            "location": district,
            "reply_text": reply_text,
            "result": {
                "district": district,
                "weather": weather,
                **level_prediction,
                "explanation": explanation,
                "water_quality": quality_result,
                "crop_advisory": crop_advisory,
            },
        }
    )


@app.route("/download-report")
def download_report():
    district = request.args.get("district")
    village = request.args.get("village")
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)

    if district and (lat is None or lng is None):
        lat, lng = get_district_coordinates(district)

    if lat is None or lng is None:
        return jsonify({"error": "Valid district or coordinates are required for report generation."}), 400

    try:
        weather = fetch_live_weather(lat, lng)
        level_prediction = predict_groundwater_level(district, weather)
        trend_rows = get_location_depth_data(district, village_name=village)
        
        quality_result = None
        crop_advisory = None
        if all(name in models for name in TARGETS):
            quality_result = run_prediction_from_features(
                level_prediction.get("features_used", {}),
                approx_depth=level_prediction.get("predicted_depth_m", 6.0),
            )
            crop_advisory = generate_crop_advisory(
                weather,
                quality_result.get("wqi"),
                level_prediction.get("predicted_depth_m", 0.0),
                quality_prediction=quality_result,
            )

        pdf_file = generate_groundwater_pdf_report(
            district=district,
            village=village,
            weather=weather,
            level_prediction=level_prediction,
            trend_rows=trend_rows,
            quality_result=quality_result,
            crop_advisory=crop_advisory,
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Could not generate report: {str(e)}"}), 500

    safe_district = (district or "location").replace(" ", "_")
    filename = f"groundwater_report_{safe_district}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(
        pdf_file,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf",
    )


@app.route("/location", methods=["GET", "POST"])
def location():
    if df.empty or "district" not in df.columns or "stn_name" not in df.columns:
        return render_template(
            "location.html",
            districts=[],
            stations=[],
            selected_district=None,
            selected_station=None,
            filtered_data=None,
        )

    districts = sorted(df["district"].dropna().unique())
    stations = []
    selected_district = selected_station = None
    filtered_data = None
    akola = next((d for d in districts if str(d).strip().lower() == "akola"), None)
    default_district = akola if akola is not None else (districts[0] if districts else None)

    if request.method == "POST":
        selected_district = request.form.get("district")
        selected_station = request.form.get("station")

        if selected_district:
            stations = sorted(
                df[df["district"] == selected_district]["stn_name"].dropna().unique()
            )

        if selected_district and selected_station:
            filtered_df = df[
                (df["district"] == selected_district) & (df["stn_name"] == selected_station)
            ].sort_values("sampling_date")

            for col in ["approx_depth", "ph", "dissolved_o2", "bod", "nitrate_n"]:
                if col in filtered_df.columns:
                    if col == "approx_depth":
                        filtered_df[col] = filtered_df[col].apply(parse_depth)
                    else:
                        filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce").fillna(0)

            if "sampling_date" in filtered_df.columns:
                filtered_df["sampling_date"] = pd.to_datetime(
                    filtered_df["sampling_date"], errors="coerce"
                ).dt.strftime("%Y-%m-%d")

            filtered_data = filtered_df.to_dict(orient="records")
    else:
        # Default selection on first open: Akola district + first station in it
        if default_district:
            selected_district = default_district
            stations = sorted(
                df[df["district"] == selected_district]["stn_name"].dropna().unique()
            )
            selected_station = stations[0] if stations else None

        if selected_district and selected_station:
            filtered_df = df[
                (df["district"] == selected_district) & (df["stn_name"] == selected_station)
            ].sort_values("sampling_date")

            for col in ["approx_depth", "ph", "dissolved_o2", "bod", "nitrate_n"]:
                if col in filtered_df.columns:
                    if col == "approx_depth":
                        filtered_df[col] = filtered_df[col].apply(parse_depth)
                    else:
                        filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce").fillna(0)

            if "sampling_date" in filtered_df.columns:
                filtered_df["sampling_date"] = pd.to_datetime(
                    filtered_df["sampling_date"], errors="coerce"
                ).dt.strftime("%Y-%m-%d")

            filtered_data = filtered_df.to_dict(orient="records")

    return render_template(
        "location.html",
        districts=districts,
        stations=stations,
        selected_district=selected_district,
        selected_station=selected_station,
        filtered_data=filtered_data,
    )


@app.route("/map")
def map_view():
    return render_template("map.html")


@app.route("/map-data")
def map_data():
    if df.empty:
        return {"data": []}

    data = []
    for _, row in df.iterrows():
        lat = convert_to_decimal(row.get("latitude"))
        lng = convert_to_decimal(row.get("longitude"))
        if not (lat and lng):
            continue

        ph = pd.to_numeric(row.get("ph"), errors="coerce")
        dissolved_o2 = pd.to_numeric(row.get("dissolved_o2"), errors="coerce")
        bod = pd.to_numeric(row.get("bod"), errors="coerce")
        nitrate_n = pd.to_numeric(row.get("nitrate_n"), errors="coerce")
        depth = parse_depth(row.get("approx_depth"))

        wqi = None
        quality = "Unknown"
        if all(pd.notna(v) for v in [ph, dissolved_o2, bod, nitrate_n]):
            wqi = calculate_wqi(ph, dissolved_o2, bod, nitrate_n)
            quality = classify_wqi(wqi)

        water_body = (
            row.get("river_body")
            or row.get("river_name")
            or row.get("water_body")
            or row.get("stn_name")
        )

        data.append(
            {
                "lat": lat,
                "lng": lng,
                "station": row.get("stn_name"),
                "district": row.get("district"),
                "water_body": water_body,
                "ph": None if pd.isna(ph) else round(float(ph), 2),
                "dissolved_o2": None if pd.isna(dissolved_o2) else round(float(dissolved_o2), 2),
                "bod": None if pd.isna(bod) else round(float(bod), 2),
                "nitrate_n": None if pd.isna(nitrate_n) else round(float(nitrate_n), 2),
                "depth": None if depth is None or pd.isna(depth) else round(float(depth), 2),
                "wqi": wqi,
                "quality": quality,
            }
        )

    return {"data": data}


@app.route("/district")
def district():
    summary = district_summary_df()
    if summary.empty:
        return render_template(
            "district.html",
            districts=[],
            depths=[],
            wqis=[],
            district_rows=[],
            total_districts=0,
            highest_wqi=0,
            lowest_wqi=0,
        )

    summary["wqi"] = pd.to_numeric(summary.get("wqi"), errors="coerce").fillna(0)
    districts = summary["district"].tolist()
    depths = summary["approx_depth"].round(2).fillna(0).tolist()
    wqis = summary["wqi"].round(2).fillna(0).tolist()
    district_rows = summary[["district", "approx_depth", "wqi", "quality"]].round(
        {"approx_depth": 2, "wqi": 2}
    ).to_dict(orient="records")

    return render_template(
        "district.html",
        districts=districts,
        depths=depths,
        wqis=wqis,
        district_rows=district_rows,
        total_districts=len(districts),
        highest_wqi=max(wqis) if wqis else 0,
        lowest_wqi=min(wqis) if wqis else 0,
    )


@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")


@app.route("/chatbot-query", methods=["POST"])
def chatbot_query():
    summary = district_summary_df()
    if summary.empty:
        return jsonify(
            {
                "reply": "I could not find dataset records right now. Please upload data from the admin panel first.",
                "follow_ups": ["How to upload dataset?", "What format should CSV have?"],
            }
        )

    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip().lower()
    if not message:
        return jsonify(
            {
                "reply": "Type your question and I will guide you. You can ask about best district, depth, comparison, or station insights.",
                "follow_ups": ["summary", "best district", "average depth district"],
            }
        )

    def format_reply(text, follow_ups=None):
        intros = [
            "Sure! ",
            "Here is what I found: ",
            "Great question. ",
            "Based on the dataset, ",
            "",
        ]
        return jsonify(
            {
                "reply": random.choice(intros) + text,
                "follow_ups": follow_ups or ["summary", "best district", "top 3 district"],
            }
        )

    # Synonym lists for intent extraction
    best_syns = ["best", "cleanest", "highest wqi", "good water", "excellent", "pure", "top quality"]
    worst_syns = ["worst", "dirtiest", "lowest wqi", "bad water", "polluted", "critical", "unfit"]
    safe_syns = ["safe", "drink", "drinking", "potable", "pure", "consumption"]
    depth_syns = ["depth", "deep", "shallow", "level", "underground", "water table", "water level"]
    summary_syns = ["summary", "overview", "stats", "statistics", "general", "total"]

    # Extract mentioned districts
    district_names = [str(d).strip() for d in summary["district"].dropna().unique().tolist()]
    matched_districts = [d for d in district_names if d.lower() in message]

    # Intent 1: Comparison of two districts (Triggered by 2 districts OR "compare/vs" with 1+ districts)
    if len(matched_districts) >= 2 or (("compare" in message or "vs" in message) and len(matched_districts) >= 1):
        if len(matched_districts) >= 2:
            d1_name, d2_name = matched_districts[0], matched_districts[1]
            d1 = summary[summary["district"] == d1_name].iloc[0]
            d2 = summary[summary["district"] == d2_name].iloc[0]
            better = d1_name if d1["wqi"] >= d2["wqi"] else d2_name
            return format_reply(
                f"Comparison result: **{d1_name}** has WQI **{d1['wqi']:.2f}** and depth **{d1['approx_depth']:.2f} m**; **{d2_name}** has WQI **{d2['wqi']:.2f}** and depth **{d2['approx_depth']:.2f} m**. Better quality district: **{better}**.",
                [f"{better} details", "top 3 district"],
            )
        else:
            return format_reply("I noticed you want to compare, but I need at least two district names. Example: *compare Pune vs Nashik*.")

    # Intent 2: Help / Guide
    if any(k in message for k in ["help", "what can you do", "commands", "guide", "how to use"]):
        return format_reply(
            "I am your AI Groundwater Assistant! You can ask me questions like:\n"
            "• *'Which district is the cleanest?'* (Best WQI)\n"
            "• *'Compare Pune and Nashik'* (District Comparison)\n"
            "• *'Is Pune water safe for drinking?'* (Safety check)\n"
            "• *'Show me the shallowest water levels'* (Depth rank)\n"
            "• *'What is WQI?'* (Glossary)",
            ["summary", "compare Pune vs Nashik", "best station"],
        )

    # Intent 3: Definition of WQI
    if "meaning of wqi" in message or "what is wqi" in message or "wqi meaning" in message or "wqi definition" in message:
        return format_reply(
            "**WQI** stands for **Water Quality Index**. It is a single number that expresses overall water quality. "
            "Higher is better: **>=80** (Excellent), **60-79** (Good), **40-59** (Poor), **<40** (Unfit for use)."
        )

    # Intent 4: Drinking Water Safety
    if any(s in message for s in safe_syns):
        # If a specific district is mentioned
        if len(matched_districts) == 1:
            d_name = matched_districts[0]
            row = summary[summary["district"] == d_name].iloc[0]
            status = "safe for drinking" if row["wqi"] >= 60 else "NOT safe for drinking without prior treatment"
            return format_reply(
                f"Based on the water quality index of **{d_name}** (WQI: **{row['wqi']:.2f}**), the water is **{status}**.",
                [f"{d_name} details", "best district"]
            )
        
        good = summary[summary["wqi"] >= 60].sort_values("wqi", ascending=False).head(5)
        if good.empty:
            return format_reply(
                "No district is currently in the Good/Excellent category in this dataset. Please treat/boil water before drinking."
            )
        names = ", ".join([f"**{r['district']}** (WQI: {r['wqi']:.2f})" for _, r in good.iterrows()])
        return format_reply(
            f"Relatively safer districts (WQI >= 60): {names}.",
            ["top 3 district", "worst district"],
        )

    # Intent 5: Summary / Overview
    if any(s in message for s in summary_syns):
        best = summary.sort_values("wqi", ascending=False).iloc[0]
        worst = summary.sort_values("wqi", ascending=True).iloc[0]
        avg_wqi = summary["wqi"].mean()
        avg_depth = summary["approx_depth"].mean()
        station_count = df["stn_name"].nunique() if "stn_name" in df.columns else 0
        return format_reply(
            f"Overview: **{len(summary)}** districts, **{station_count}** stations, average WQI **{avg_wqi:.2f}**, average depth **{avg_depth:.2f} m**. "
            f"Best district: **{best['district']}** ({best['wqi']:.2f}). Most critical: **{worst['district']}** ({worst['wqi']:.2f}).",
            ["best district", "worst district", "average depth district"],
        )

    # Intent 6: Top 3 / Best performers
    if "top 3" in message or "top three" in message or "best 3" in message:
        top = summary.sort_values("wqi", ascending=False).head(3)
        items = [f"**{row['district']}** ({row['wqi']:.2f})" for _, row in top.iterrows()]
        return format_reply("Top 3 districts by WQI: " + ", ".join(items), ["worst district", "summary"])

    # Intent 7: Shallowest / Lowest Depth
    if "lowest depth" in message or "shallow" in message:
        shallow = summary.sort_values("approx_depth", ascending=True).head(3)
        items = [f"**{row['district']}** ({row['approx_depth']:.2f} m)" for _, row in shallow.iterrows()]
        return format_reply("Shallowest groundwater depth districts: " + ", ".join(items))

    # Intent 8: Deepest / Highest Depth
    if "highest depth" in message or "deep" in message:
        deep = summary.sort_values("approx_depth", ascending=False).head(3)
        items = [f"**{row['district']}** ({row['approx_depth']:.2f} m)" for _, row in deep.iterrows()]
        return format_reply("Deepest groundwater depth districts: " + ", ".join(items))

    # Intent 9: Best Station
    if "station" in message and ("best" in message or "top" in message):
        if {"district", "stn_name", "ph", "dissolved_o2", "bod", "nitrate_n"}.issubset(df.columns):
            station_df = df.copy()
            for col in ["ph", "dissolved_o2", "bod", "nitrate_n"]:
                station_df[col] = pd.to_numeric(station_df[col], errors="coerce")
            station_df = station_df.dropna(subset=["district", "stn_name", "ph", "dissolved_o2", "bod", "nitrate_n"])
            if not station_df.empty:
                station_group = (
                    station_df.groupby(["district", "stn_name"])[["ph", "dissolved_o2", "bod", "nitrate_n"]]
                    .mean()
                    .reset_index()
                )
                station_group["wqi"] = station_group.apply(
                    lambda r: calculate_wqi(r["ph"], r["dissolved_o2"], r["bod"], r["nitrate_n"]),
                    axis=1,
                )
                best_stations = station_group.sort_values("wqi", ascending=False).head(3)
                items = [
                    f"**{row['stn_name']}** ({row['district']}, WQI: {row['wqi']:.2f})"
                    for _, row in best_stations.iterrows()
                ]
                return format_reply("Top stations by water quality: " + ", ".join(items))

    # Intent 10: River / Water Body
    if "river" in message or "water body" in message or "lake" in message:
        if "water_body" in df.columns or "river_name" in df.columns or "river_body" in df.columns:
            col = "water_body" if "water_body" in df.columns else ("river_name" if "river_name" in df.columns else "river_body")
            top_bodies = (
                df[col].dropna().astype(str).str.strip().value_counts().head(5).index.tolist()
            )
            if top_bodies:
                return format_reply("Top tracked water bodies in dataset: " + ", ".join(top_bodies))

    # Intent 11: Best District
    if any(b in message for b in best_syns):
        best = summary.sort_values("wqi", ascending=False).iloc[0]
        session["chat_last_district"] = str(best["district"])
        return format_reply(
            f"The best district is **{best['district']}** with average WQI **{best['wqi']:.2f}** ({best['quality']}).",
            [f"{best['district']} details", "compare district"],
        )

    # Intent 12: Worst District
    if any(w in message for w in worst_syns):
        worst = summary.sort_values("wqi", ascending=True).iloc[0]
        session["chat_last_district"] = str(worst["district"])
        return format_reply(
            f"The most critical/worst district is **{worst['district']}** with average WQI **{worst['wqi']:.2f}** ({worst['quality']}).",
            [f"{worst['district']} details", "safe drinking water"],
        )

    # Intent 13: Average Depth
    if any(d in message for d in depth_syns):
        # If specific district is mentioned
        if len(matched_districts) == 1:
            d_name = matched_districts[0]
            row = summary[summary["district"] == d_name].iloc[0]
            return format_reply(
                f"The average groundwater depth in **{d_name}** is **{row['approx_depth']:.2f} m**.",
                [f"{d_name} details", "average depth district"]
            )
        
        avg_depth = summary["approx_depth"].mean()
        return format_reply(f"Overall district average groundwater depth is **{avg_depth:.2f} m**.")

    # Intent 14: Contextual follow-up ("tell more", "its details")
    if ("its details" in message or "tell more" in message or "more details" in message) and session.get("chat_last_district"):
        matched_districts = [session.get("chat_last_district", "")]

    # Intent 15: Single District Lookup (matches Pune, Nashik, etc.)
    if len(matched_districts) == 1:
        district_name = matched_districts[0]
        row = summary[summary["district"] == district_name].iloc[0]
        suggestion = "Recommended for routine use." if row["wqi"] >= 60 else "Needs treatment before domestic use."
        session["chat_last_district"] = str(district_name)
        return format_reply(
            f"**{district_name}**: Average depth **{row['approx_depth']:.2f} m**, average WQI **{row['wqi']:.2f}** ({row['quality']}). *{suggestion}*",
            ["compare district", "safe drinking water"],
        )

    # Default fallback
    return format_reply(
        "I didn't quite catch that. Try asking in natural style, like:\n"
        "• *'Which district is best?'*\n"
        "• *'Is Pune water safe?'*\n"
        "• *'Compare Pune and Nashik'*",
        ["summary", "best district", "help"],
    )


@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "admin123":
            session["admin"] = True
            return redirect("/admin/dashboard")
        return render_template("admin_login.html", error="Invalid Login")

    return render_template("admin_login.html")


@app.route("/admin/dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect("/admin")

    rows = len(df) if not df.empty else 0
    districts = df["district"].nunique() if not df.empty and "district" in df.columns else 0
    message = request.args.get("message")
    error = request.args.get("error")
    return render_template(
        "admin_dashboard.html",
        rows=rows,
        districts=districts,
        message=message,
        error=error,
    )


def _clean_uploaded_data(df_new):
    df_new = df_new.dropna(how="all")
    for col in NUMERIC_COLS:
        if col in df_new.columns:
            if col == "approx_depth":
                df_new[col] = df_new[col].apply(parse_depth)
            else:
                df_new[col] = pd.to_numeric(df_new[col], errors="coerce")
    if "sampling_date" in df_new.columns:
        df_new["sampling_date"] = pd.to_datetime(df_new["sampling_date"], errors="coerce")
    return df_new


@app.route("/admin/upload", methods=["POST"])
@app.route("/admin/confirm-upload", methods=["POST"])
def confirm_upload():
    global df
    if "admin" not in session:
        return redirect("/admin")

    file = request.files.get("dataset")
    if not file:
        return redirect("/admin/dashboard?error=No+file+selected")
    if not file.filename.lower().endswith(".csv"):
        return redirect("/admin/dashboard?error=Please+upload+a+CSV+file")

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    try:
        uploaded_raw = pd.read_csv(path)
    except Exception:
        return redirect("/admin/dashboard?error=Uploaded+CSV+could+not+be+read")

    uploaded_clean = _clean_uploaded_data(uploaded_raw)

    missing_required = [col for col in FEATURES + TARGETS if col not in uploaded_clean.columns]
    if missing_required:
        return redirect(
            "/admin/dashboard?error=Missing+required+columns:+"
            + ",".join(missing_required[:6])
        )

    base_df = df.copy() if not df.empty else pd.DataFrame(columns=uploaded_clean.columns)
    if not base_df.empty:
        base_df = _clean_uploaded_data(base_df)

    merged_df = pd.concat([base_df, uploaded_clean], ignore_index=True, sort=False)

    dedupe_cols = [c for c in ["district", "stn_name", "sampling_date"] if c in merged_df.columns]
    if dedupe_cols:
        merged_df = merged_df.drop_duplicates(subset=dedupe_cols, keep="last")

    merged_df = merged_df.dropna(subset=FEATURES + TARGETS)
    if merged_df.empty:
        return redirect("/admin/dashboard?error=No+valid+rows+available+after+cleaning")

    merged_df.to_csv("groundwater_cleaned.csv", index=False)
    df = merged_df

    train_models(df)
    return redirect(
        "/admin/dashboard?message=Dataset+updated+successfully.+Models+retrained+with+latest+months."
    )


def train_models(data):
    global models
    for target in TARGETS:
        if target in data.columns and all(col in data.columns for col in FEATURES):
            X = data[FEATURES]
            y = data[target]
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            filename = f"{target}_model.pkl"
            joblib.dump(model, filename)
            models[target] = model
            print(target, "model retrained")

    if DEPTH_TARGET in data.columns and all(col in data.columns for col in FEATURES):
        depth_df = data.copy()
        for col in FEATURES:
            depth_df[col] = pd.to_numeric(depth_df[col], errors="coerce")
        depth_df[DEPTH_TARGET] = depth_df[DEPTH_TARGET].apply(parse_depth)
        depth_df = depth_df.dropna(subset=FEATURES + [DEPTH_TARGET])
        if not depth_df.empty:
            X = depth_df[FEATURES]
            y = depth_df[DEPTH_TARGET]
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
            depth_model = RandomForestRegressor()
            depth_model.fit(X_train, y_train)
            filename = f"{DEPTH_TARGET}_model.pkl"
            joblib.dump(depth_model, filename)
            models[DEPTH_TARGET] = depth_model
            print(DEPTH_TARGET, "model retrained")


def train_depth_model_only(data):
    """
    Train/refresh only the depth model using `parse_depth` (for categorical depth values).
    """
    global models
    if data is None or data.empty:
        return
    if DEPTH_TARGET not in data.columns:
        return
    if not all(col in data.columns for col in FEATURES):
        return

    depth_df = data.copy()
    for col in FEATURES:
        depth_df[col] = pd.to_numeric(depth_df[col], errors="coerce")
    depth_df[DEPTH_TARGET] = depth_df[DEPTH_TARGET].apply(parse_depth)
    depth_df = depth_df.dropna(subset=FEATURES + [DEPTH_TARGET])

    if depth_df.empty:
        return

    X = depth_df[FEATURES]
    y = depth_df[DEPTH_TARGET]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
    depth_model = RandomForestRegressor()
    depth_model.fit(X_train, y_train)

    filename = f"{DEPTH_TARGET}_model.pkl"
    joblib.dump(depth_model, filename)
    models[DEPTH_TARGET] = depth_model
    print(DEPTH_TARGET, "model retrained (depth only)")


if not df.empty and DEPTH_TARGET in df.columns:
    # Dataset depth is categorical (e.g., 'Less than 50cm'); ensure depth model is trained correctly.
    sample = df[DEPTH_TARGET].dropna().astype(str).head(50)
    depth_is_categorical = sample.str.contains(
        r"cm|greater than|less than|no information|no info",
        case=False,
        regex=True,
    ).any()

    if depth_is_categorical:
        try:
            train_depth_model_only(df)
        except Exception:
            pass
    elif DEPTH_TARGET not in models:
        try:
            train_models(df)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(debug=True)
