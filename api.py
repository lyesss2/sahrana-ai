"""
Sahrana API Backend
===================
Flask REST API that serves the trained ML model (sahrana_model.pkl).
 
Endpoints:
  POST /api/predict   — run a prediction
  GET  /api/health    — health check
  GET  /api/crops     — list supported crops
"""
 
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, os, pandas as pd
 
app = Flask(__name__)
CORS(app)
 
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sahrana_model.pkl")
data = joblib.load(MODEL_PATH)
 
model_crop         = data["model_crop"]
model_irrigation   = data["model_irrigation"]
model_efficiency   = data["model_efficiency"]
water_encoder      = data["water_encoder"]
soil_encoder       = data["soil_encoder"]
crop_encoder       = data["crop_encoder"]
irrigation_encoder = data["irrigation_encoder"]
efficiency_encoder = data["efficiency_encoder"]
 
VALID_SOILS = list(soil_encoder.classes_)
VALID_WATER = list(water_encoder.classes_)
VALID_CROPS = list(crop_encoder.classes_)
 
def do_predict(temperature_c, peak_solar_hours, water_availability, soil_type, humidity_pct):
    water_num = water_encoder.transform([water_availability])[0]
    soil_num  = soil_encoder.transform([soil_type])[0]
    row = pd.DataFrame(
        [[temperature_c, peak_solar_hours, water_num, soil_num, humidity_pct]],
        columns=["temperature_c","peak_solar_hours","water_encoded","soil_encoded","humidity_pct"]
    )
    return {
        "recommended_crop":    crop_encoder.inverse_transform(model_crop.predict(row))[0],
        "irrigation_timing":   irrigation_encoder.inverse_transform(model_irrigation.predict(row))[0],
        "resource_efficiency": efficiency_encoder.inverse_transform(model_efficiency.predict(row))[0],
    }
 
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "sahrana_model.pkl"})
 
@app.route("/api/crops")
def crops():
    return jsonify({"crops": VALID_CROPS, "count": len(VALID_CROPS)})
 
@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True) or {}
 
    temperature_c      = body.get("temperature_c")      if body.get("temperature_c")      is not None else body.get("temperature")
    peak_solar_hours   = body.get("peak_solar_hours")   if body.get("peak_solar_hours")   is not None else body.get("solar")
    water_availability = body.get("water_availability") if body.get("water_availability") is not None else body.get("water")
    soil_type          = body.get("soil_type")          if body.get("soil_type")          is not None else body.get("soil")
    humidity_pct       = body.get("humidity_pct")       if body.get("humidity_pct")       is not None else body.get("humidity")
 
    missing = [k for k,v in {
        "temperature_c": temperature_c, "peak_solar_hours": peak_solar_hours,
        "water_availability": water_availability, "soil_type": soil_type,
        "humidity_pct": humidity_pct
    }.items() if v is None]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
 
    soil_type = str(soil_type).lower().strip()
    if soil_type not in VALID_SOILS:
        if "loam" in soil_type:   soil_type = "loamy"
        elif "silt" in soil_type: soil_type = "silty"
        elif "clay" in soil_type: soil_type = "clay"
        else:                     soil_type = "sandy"
 
    water_availability = str(water_availability).lower().strip()
    if water_availability not in VALID_WATER:
        water_availability = "medium"
 
    try:
        result = do_predict(float(temperature_c), float(peak_solar_hours),
                            water_availability, soil_type, float(humidity_pct))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
if __name__ == "__main__":
    print("Sahrana API  →  http://127.0.0.1:5000")
    print(f"Valid soils: {VALID_SOILS}")
    print(f"Valid water: {VALID_WATER}")
    app.run(debug=True, port=5000)
 
