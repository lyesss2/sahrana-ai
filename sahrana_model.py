"""
sahrana_model.py
----------------
Trains a Random Forrest + a Decision Tree to predict:
  1. Which crop to plant  = the main output.
  2. When to irrigate.
  3. How efficient the resource usage is.

"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

#1: Load the dataset
# pandas read our CSV into a DataFrame.
df = pd.read_csv("sahrana_dataset.csv")
print(f"Loaded {len(df)} rows")
print(df.head(3))
print()

#2: Encode text columns into numbers
# The model only understands numbers, not words like sandy or low.
# LabelEncoder converts: "clay"to 0, "loamy"to 1, "sandy"to 2, "silty"to 3
water_encoder = LabelEncoder()
soil_encoder  = LabelEncoder()

df["water_encoded"] = water_encoder.fit_transform(df["water_availability"])
df["soil_encoded"]  = soil_encoder.fit_transform(df["soil_type"])

#3: Define inputs (X) and outputs (y)
# X = the 5 factors the model reads
# y = what we want it to predict (one y per output)
X = df[["temperature_c", "peak_solar_hours", "water_encoded", "soil_encoded", "humidity_pct"]]

crop_encoder       = LabelEncoder()
irrigation_encoder = LabelEncoder()
efficiency_encoder = LabelEncoder()

y_crop       = crop_encoder.fit_transform(df["crop"])
y_irrigation = irrigation_encoder.fit_transform(df["irrigation_timing"])
y_efficiency = efficiency_encoder.fit_transform(df["resource_efficiency"])

#4: Split into training and testing sets
#We hide 20% of rows so we can test on data the model has never seen.
X_train, X_test, yc_train, yc_test, yi_train, yi_test, ye_train, ye_test = train_test_split(
    X, y_crop, y_irrigation, y_efficiency,
    test_size=0.2,
    random_state=42,
    stratify=y_crop
)
print(f"Training rows: {len(X_train)},  Testing rows: {len(X_test)}")
print()

#5: Train the models
#DecisionTreeClassifier builds a tree of if/else rules from the data.
#Easier for rule-based modeling.
#max_depth limits how deep the tree goes (avoids memorising instead of learning).


model_irrigation = DecisionTreeClassifier(max_depth=5,  random_state=42)
model_efficiency = DecisionTreeClassifier(max_depth=4,  random_state=42)

#Updating only the crop model to random forest for better accuracy

from sklearn.ensemble import RandomForestClassifier
model_crop = RandomForestClassifier(
    n_estimators=500,
    max_depth=7,
    random_state=42,
    class_weight='balanced'
)


model_crop.fit(X_train, yc_train)        # .fit() is where learning happens
model_irrigation.fit(X_train, yi_train)
model_efficiency.fit(X_train, ye_train)
print("Models trained!")

#6: Check accuracy on the test set
crop_acc = accuracy_score(yc_test, model_crop.predict(X_test))
irr_acc  = accuracy_score(yi_test, model_irrigation.predict(X_test))
eff_acc  = accuracy_score(ye_test, model_efficiency.predict(X_test))

print(f"Crop recommendation accuracy:  {crop_acc:.0%}")
print(f"Irrigation timing accuracy:    {irr_acc:.0%}")
print(f"Resource efficiency accuracy:  {eff_acc:.0%}")
print()

#7: Save everything to one file so the dashboard can load it
joblib.dump({
    "model_crop":          model_crop,
    "model_irrigation":    model_irrigation,
    "model_efficiency":    model_efficiency,
    "water_encoder":       water_encoder,
    "soil_encoder":        soil_encoder,
    "crop_encoder":        crop_encoder,
    "irrigation_encoder":  irrigation_encoder,
    "efficiency_encoder":  efficiency_encoder,
}, "sahrana_model.pkl")
print("Model saved to sahrana_model.pkl")
print()

#8: Prediction function : what the dashboard will call
def predict_crop(temperature_c, peak_solar_hours, water_availability, soil_type, humidity_pct):
    #Encode text inputs to numbers (same way as during training)
    water_num = water_encoder.transform([water_availability])[0]
    soil_num  = soil_encoder.transform([soil_type])[0]

    #Build one-row table (bcs the model expects a table, not just values)
    input_row = pd.DataFrame(
        [[temperature_c, peak_solar_hours, water_num, soil_num, humidity_pct]],
        columns=["temperature_c", "peak_solar_hours", "water_encoded", "soil_encoded", "humidity_pct"]
    )

    #Ask each model to predict, then decode numbers back to text
    return {
        "recommended_crop":    crop_encoder.inverse_transform(model_crop.predict(input_row))[0], #gives us the text.
        "irrigation_timing":   irrigation_encoder.inverse_transform(model_irrigation.predict(input_row))[0],
        "resource_efficiency": efficiency_encoder.inverse_transform(model_efficiency.predict(input_row))[0],
    }


#Try it
result = predict_crop(
    temperature_c=42,
    peak_solar_hours=10,
    water_availability="low",
    soil_type="sandy",
    humidity_pct=15,
)
print("Test prediction (42C, sandy, low water, 10 solar hours, 15% humidity):")
print(f"  Recommended crop:    {result['recommended_crop']}")
print(f"  Irrigation timing:   {result['irrigation_timing']}")
print(f"  Resource efficiency: {result['resource_efficiency']}")

