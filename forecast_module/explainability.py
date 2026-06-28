import numpy as np
import pandas as pd

HAS_SHAP = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    pass

FEATURES_CLEAN_NAMES = {
    "depth_lag_1": "Previous Water Table Depth",
    "depth_lag_2": "Depth (Lag 2d)",
    "depth_lag_3": "Depth (Lag 3d)",
    "rainfall_lag_1": "Rainfall (Lag 1d)",
    "rainfall_lag_2": "Rainfall (Lag 2d)",
    "rainfall_lag_3": "Rainfall (Lag 3d)",
    "temperature_lag_1": "Temperature (Lag 1d)",
    "temperature_lag_2": "Temperature (Lag 2d)",
    "temperature_lag_3": "Temperature (Lag 3d)",
    "humidity_lag_1": "Humidity (Lag 1d)",
    "humidity_lag_2": "Humidity (Lag 2d)",
    "humidity_lag_3": "Humidity (Lag 3d)",
    "month_sin": "Seasonal (Month Sin)",
    "month_cos": "Seasonal (Month Cos)"
}


def explain_prediction(model_name, best_model, historical_df, last_features_row):
    """
    Computes SHAP values or feature importance mapping to identify 
    how parameters pulled predictions up or down relative to baseline averages.
    """
    # Preprocess feature names
    features = list(last_features_row.keys())
    values = [float(last_features_row[f]) for f in features]
    
    # Calculate baseline (means)
    baselines = {}
    for f in features:
        if f in historical_df.columns:
            baselines[f] = float(historical_df[f].mean())
        else:
            baselines[f] = 0.0

    contributions = []

    shap_model = best_model
    if model_name == "Weighted Ensemble":
        # Extract the Random Forest model from the ensemble for SHAP
        for sub_m, weight in best_model.models_with_weights:
            if hasattr(sub_m, "feature_importances_") and not hasattr(sub_m, "learning_rate"):
                shap_model = sub_m
                break

    # 1. SHAP values (if importable and model supports tree explainer)
    if HAS_SHAP and (model_name in ["Random Forest", "XGBoost"] or model_name == "Weighted Ensemble") and hasattr(shap_model, "predict"):
        try:
            # Create subset training sample for background
            bg_data = historical_df[features].dropna().head(100).values
            explainer = shap.TreeExplainer(shap_model)
            shap_values = explainer.shap_values(np.array([values]))
            
            # Unpack array
            if isinstance(shap_values, list):
                # For classification/multiple outputs
                shap_flat = shap_values[0].flatten()
            else:
                shap_flat = shap_values.flatten()
                
            for f, val, shap_val in zip(features, values, shap_flat):
                clean_name = FEATURES_CLEAN_NAMES.get(f, f.replace("_", " ").title())
                contributions.append({
                    "feature": f,
                    "label": clean_name,
                    "value": val,
                    "baseline": baselines.get(f, 0.0),
                    "impact": float(shap_val),
                    "direction": "increase" if shap_val >= 0 else "decrease"
                })
        except Exception:
            pass

    # 2. Fallback: Heuristic feature importance and deviation-based impact calculation
    if not contributions:
        # Calculate feature importances from model if available
        importances = {}
        if model_name == "Weighted Ensemble":
            for sub_m, weight in best_model.models_with_weights:
                if hasattr(sub_m, "feature_importances_"):
                    for f, imp in zip(features, sub_m.feature_importances_):
                        importances[f] = importances.get(f, 0.0) + weight * float(imp)
        elif hasattr(best_model, "feature_importances_"):
            importances = {f: float(imp) for f, imp in zip(features, best_model.feature_importances_)}
        elif hasattr(best_model, "coef_"):
            # Linear model coefficients
            coef_total = sum(abs(float(c)) for c in best_model.coef_)
            if coef_total > 0:
                importances = {f: abs(float(c)) / coef_total for f, c in zip(features, best_model.coef_)}
                
        # Default heuristic weights if no importances available
        if not importances:
            importances = {
                "depth_lag_1": 0.45, "depth_lag_2": 0.15, "depth_lag_3": 0.05,
                "rainfall_lag_1": 0.15, "rainfall_lag_2": 0.05, "rainfall_lag_3": 0.02,
                "temperature_lag_1": 0.04, "temperature_lag_2": 0.02, "temperature_lag_3": 0.01,
                "humidity_lag_1": 0.03, "humidity_lag_2": 0.02, "humidity_lag_3": 0.01,
                "month_sin": 0.02, "month_cos": 0.03
            }
            
        for f, val in zip(features, values):
            base = baselines.get(f, 0.0)
            importance = importances.get(f, 0.02)
            
            # Deviation ratio
            if abs(base) > 0.01:
                diff_pct = (val - base) / base
            else:
                diff_pct = val - base
                
            # Impact is proportional to feature importance * normalized deviation
            # Rainfall and humidity increases pull depth DOWN (recharge = depth decrease)
            # Temperature and historical high depth values pull depth UP (depletion = depth increase)
            direction_factor = 1.0
            if "rainfall" in f or "humidity" in f:
                direction_factor = -1.0
                
            impact = diff_pct * importance * direction_factor
            clean_name = FEATURES_CLEAN_NAMES.get(f, f.replace("_", " ").title())
            
            contributions.append({
                "feature": f,
                "label": clean_name,
                "value": val,
                "baseline": base,
                "impact": float(impact),
                "direction": "increase" if impact >= 0 else "decrease"
            })
            
    # Sort contributions by absolute impact (highest first)
    contributions = sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)
    
    # Generate high-fidelity clean strings explaining factors (top 5 factors)
    explanations = []
    for c in contributions[:5]:
        val_diff = c["value"] - c["baseline"]
        base_val = c["baseline"]
        
        # Calculate percentage difference
        if abs(base_val) > 0.01:
            diff_pct = int(round((val_diff / base_val) * 100))
        else:
            diff_pct = int(round(val_diff * 100))
            
        direction_word = "increased" if val_diff >= 0 else "decreased"
        sign_str = "+" if val_diff >= 0 else "-"
        
        if "depth" in c["feature"]:
            explanations.append(f"{c['label']} was {direction_word} ({sign_str}{abs(diff_pct)}%) relative to average")
        elif "rainfall" in c["feature"]:
            explanations.append(f"Rainfall was {direction_word} ({sign_str}{abs(diff_pct)}%) relative to mean")
        elif "temperature" in c["feature"]:
            explanations.append(f"Temperature was {direction_word} ({sign_str}{abs(diff_pct)}%) relative to mean")
        elif "humidity" in c["feature"]:
            explanations.append(f"Humidity was {direction_word} ({sign_str}{abs(diff_pct)}%) relative to mean")
        else:
            explanations.append(f"Seasonal factor shifts ({sign_str}{abs(diff_pct)}%)")

    return {
        "contributions": contributions,
        "summary_points": explanations
    }
