#making sure we load everything from the saved model

import joblib
import pandas as pd

data = joblib.load("sahrana_model.pkl")

model_crop = data["model_crop"]
model_irrigation = data["model_irrigation"]
model_efficiency = data["model_efficiency"]

water_encoder = data["water_encoder"]
soil_encoder = data["soil_encoder"]
crop_encoder = data["crop_encoder"]
irrigation_encoder = data["irrigation_encoder"]
efficiency_encoder = data["efficiency_encoder"]


def predict_crop(temperature_c, peak_solar_hours, water_availability, soil_type, humidity_pct):
    water_num = water_encoder.transform([water_availability])[0]
    soil_num  = soil_encoder.transform([soil_type])[0]

    input_row = pd.DataFrame(
        [[temperature_c, peak_solar_hours, water_num, soil_num, humidity_pct]],
        columns=["temperature_c", "peak_solar_hours", "water_encoded", "soil_encoded", "humidity_pct"]
    )

    return {
        "recommended_crop": crop_encoder.inverse_transform(model_crop.predict(input_row))[0],
        "irrigation_timing": irrigation_encoder.inverse_transform(model_irrigation.predict(input_row))[0],
        "resource_efficiency": efficiency_encoder.inverse_transform(model_efficiency.predict(input_row))[0],
    }



    # Example prediction
if __name__ == "__main__":
    result = predict_crop(
        temperature_c=42,
        peak_solar_hours=10,
        water_availability="low",
        soil_type="sandy",
        humidity_pct=15
    )
    print("Test prediction:")
    print(f"Recommended crop:    {result['recommended_crop']}")
    print(f"Irrigation timing:   {result['irrigation_timing']}")
    print(f"Resource efficiency: {result['resource_efficiency']}")
