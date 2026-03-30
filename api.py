
"""
api.py — Sahrana Flask REST API
================================
Loads sahrana_model.pkl once at startup and serves predictions.

Endpoints
---------
POST /api/predict    — main prediction
GET  /api/options    — valid input values for frontend dropdowns
GET  /api/health     — uptime check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

#  Load model bundle once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sahrana_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "sahrana_model.pkl not found. Run sahrana_model.py first to train and save the model."
    )

bundle             = joblib.load(MODEL_PATH)
model_crop         = bundle["model_crop"]
model_irrigation   = bundle["model_irrigation"]
model_efficiency   = bundle["model_efficiency"]
water_encoder      = bundle["water_encoder"]
soil_encoder       = bundle["soil_encoder"]
crop_encoder       = bundle["crop_encoder"]
irrigation_encoder = bundle["irrigation_encoder"]
efficiency_encoder = bundle["efficiency_encoder"]

FEATURES = ["temperature_c", "peak_solar_hours", "water_encoded", "soil_encoded", "humidity_pct"]

print("✅  Model loaded successfully")
print(f"    Crops      : {list(crop_encoder.classes_)}")
print(f"    Water lvls : {list(water_encoder.classes_)}")
print(f"    Soil types : {list(soil_encoder.classes_)}")


#  Validation 
def validate(data):
    errors = []
    required = ["temperature_c", "peak_solar_hours", "water_availability", "soil_type", "humidity_pct"]
    for f in required:
        if f not in data:
            errors.append(f"Missing field: '{f}'")
    if errors:
        return errors
    if data["water_availability"] not in list(water_encoder.classes_):
        errors.append(f"water_availability must be one of {list(water_encoder.classes_)}")
    if data["soil_type"] not in list(soil_encoder.classes_):
        errors.append(f"soil_type must be one of {list(soil_encoder.classes_)}")
    try:
        t = float(data["temperature_c"])
        if not (0 <= t <= 60):
            errors.append("temperature_c must be between 0 and 60")
    except (ValueError, TypeError):
        errors.append("temperature_c must be a number")
    try:
        s = float(data["peak_solar_hours"])
        if not (0 <= s <= 14):
            errors.append("peak_solar_hours must be between 0 and 14")
    except (ValueError, TypeError):
        errors.append("peak_solar_hours must be a number")
    try:
        h = float(data["humidity_pct"])
        if not (0 <= h <= 100):
            errors.append("humidity_pct must be between 0 and 100")
    except (ValueError, TypeError):
        errors.append("humidity_pct must be a number")
    return errors


# Core inference 
def run_prediction(temperature_c, peak_solar_hours, water_availability, soil_type, humidity_pct):
    water_num = water_encoder.transform([water_availability])[0]
    soil_num  = soil_encoder.transform([soil_type])[0]

    input_row = pd.DataFrame(
        [[temperature_c, peak_solar_hours, water_num, soil_num, humidity_pct]],
        columns=FEATURES
    )

    crop_proba = model_crop.predict_proba(input_row)[0]
    crop_idx   = int(crop_proba.argmax())
    crop_label = crop_encoder.inverse_transform([crop_idx])[0]
    confidence = round(float(crop_proba[crop_idx]) * 100, 1)

    top3_idx = crop_proba.argsort()[::-1][:3]
    top3 = [
        {"crop": crop_encoder.inverse_transform([int(i)])[0],
         "confidence": round(float(crop_proba[i]) * 100, 1)}
        for i in top3_idx
    ]

    irrigation = irrigation_encoder.inverse_transform(
        model_irrigation.predict(input_row)
    )[0]
    efficiency = efficiency_encoder.inverse_transform(
        model_efficiency.predict(input_row)
    )[0]

    return {
        "recommended_crop":    crop_label,
        "confidence_pct":      confidence,
        "top3_alternatives":   top3,
        "irrigation_timing":   irrigation,
        "resource_efficiency": efficiency,
    }


#  Routes
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "sahrana_model.pkl"})


@app.route("/api/options")
def options():
    return jsonify({
        "water_availability": list(water_encoder.classes_),
        "soil_type":          list(soil_encoder.classes_),
        "crops":              list(crop_encoder.classes_),
        "temperature_range":  [0, 60],
        "solar_range":        [0, 14],
        "humidity_range":     [0, 100],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    errors = validate(data)
    if errors:
        return jsonify({"error": "Invalid inputs", "details": errors}), 400
    try:
        result = run_prediction(
            temperature_c      = float(data["temperature_c"]),
            peak_solar_hours   = float(data["peak_solar_hours"]),
            water_availability = data["water_availability"],
            soil_type          = data["soil_type"],
            humidity_pct       = float(data["humidity_pct"]),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


# Run 
if __name__ == "__main__":
    print("\n🌾  Sahrana API starting on http://localhost:5000")
    print("    POST /api/predict")
    print("    GET  /api/options")
    print("    GET  /api/health\n")
    app.run(debug=True, port=5000)
