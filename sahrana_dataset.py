"""
sahrana_dataset.py
------------------
Creates a sample (synthetic) dataset for training our crop prediction model.
Since we don't have real Algerian farm data yet, we make up realistic numbers
based on what we know about each crop's needs.

"""

import random
import csv

#Fix the random seed so we get the same dataset every time we run this
random.seed(42)


#1: Define what conditions each crop grows best in

# Each crop has:
#temp_min / temp_max : temperature range in Celsius it likes
#solar_min / solar_max : peak sun hours per day it needs
#water : how much water it needs (low/medium/high)
#soils : list of soil types it can grow in
#humidity_min/max : humidity % range it tolerates


CROP_CONDITIONS = [
    {
        "crop":  "Dates (Palm)",
        "temp_min":     36, "temp_max":  48,   #loves extreme heat
        "solar_min":    8,  "solar_max": 12,
        "water":        ["low", "medium"],      #drought-tolerant
        "soils":        ["sandy", "loamy"],
        "humidity_min": 10, "humidity_max": 35, #prefers dry air
    },
    {
        "crop":    "Wheat",
        "temp_min":     8,  "temp_max":  22,    #cool season crop
        "solar_min":    5,  "solar_max": 8,
        "water":        ["medium", "high"],
        "soils":        ["loamy", "clay", "silty"],
        "humidity_min": 30, "humidity_max": 65,
    },
    {
        "crop":    "Barley",
        "temp_min":     7,  "temp_max":  20,    #even more cold tolerant than wheat
        "solar_min":    4,  "solar_max": 8,
        "water":        ["low", "medium"],       #more drought-resistant than wheat
        "soils":        ["loamy", "sandy", "silty"],
        "humidity_min": 25, "humidity_max": 60,
    },
    {
        "crop":    "Sorghum",     
        "temp_min":     24, "temp_max":  38,
        "solar_min":    7,  "solar_max": 11,
        "water":        ["low", "medium"],       #very drought-resistant
        "soils":        ["sandy", "loamy", "clay"],
        "humidity_min": 20, "humidity_max": 50,
    },
    {
        "crop":    "Watermelon",
        "temp_min":     25, "temp_max":  35,     #loves medium heat
        "solar_min":    7,  "solar_max": 11,
        "water":        ["medium", "high"],
        "soils":        ["sandy", "loamy"],
        "humidity_min": 30, "humidity_max": 60,
    },
    {
        "crop":     "Tomatoes",
        "temp_min":     18, "temp_max":  30,
        "solar_min":    6,  "solar_max": 10,
        "water":        ["medium", "high"],
        "soils":        ["loamy", "clay"],
        "humidity_min": 40, "humidity_max": 70,
    },
    {
        "crop":      "Corn",
        "temp_min":     20, "temp_max":  32,
        "solar_min":    6,  "solar_max": 10,
        "water":        ["high"],               #needs a lot of water
        "soils":        ["loamy", "clay", "silty"],
        "humidity_min": 40, "humidity_max": 75,
    },
    {
        "crop":      "Olives",
        "temp_min":     15, "temp_max":  35,    #wide range, very hardy
        "solar_min":    7,  "solar_max": 11,
        "water":        ["low", "medium"],
        "soils":        ["loamy", "sandy"],
        "humidity_min": 20, "humidity_max": 55,
    },
    {
        "crop":      "Peppers",
        "temp_min":     20, "temp_max":  32,
        "solar_min":    6,  "solar_max": 10,
        "water":        ["medium", "high"],
        "soils":        ["loamy", "clay"],
        "humidity_min": 40, "humidity_max": 70,
    },
    {
        "crop":        "Rice",
        "temp_min":     20, "temp_max":  35,
        "solar_min":    6,  "solar_max": 10,
        "water":        ["high"],               # needs the most water of all
        "soils":        ["clay", "silty"],      # needs water-holding soil
        "humidity_min": 60, "humidity_max": 90,
    },
]

#2: Rules for the two extra outputs
# Irrigation timing depends only on water_availability
# (more scarce water = water less often)

def get_irrigation(water):
    if water == "low":
        return "every 7-10 days"
    elif water == "medium":
        return "every 3-5 days"
    else:  # high
        return "every 1-2 days"

#Resource efficiency: are we using water wisely?
#Low water need + lots of sun (solar energy available) = high efficiency

def get_efficiency(water, solar_hours):
    if water == "low" and solar_hours >= 8:  #less needed water level = efficient
        return "high"
    elif water == "high":  #needs water a lot = not efficient
        return "low"
    else:
        return "medium"  #medium 

#3: Generate rows

# For each crop, we create 200 sample data rows.
# Each row 'randomly' picks values within that crop's comfort zone,
# with a small random variation so the model doesn't overfit.

ROWS_PER_CROP = 200
rows = []

for crop_info in CROP_CONDITIONS:
    for _ in range(ROWS_PER_CROP):

        #random temperature within this crop's range
        temp = round(random.uniform(crop_info["temp_min"], crop_info["temp_max"]), 1)

        #random solar hours
        solar = round(random.uniform(crop_info["solar_min"], crop_info["solar_max"]), 1)

        #random water level from the crop's allowed list!
        water = random.choice(crop_info["water"])

        #random soil type from the crop's allowed list
        soil = random.choice(crop_info["soils"])

        #random humidity within the crop's range
        humidity = round(random.uniform(crop_info["humidity_min"], crop_info["humidity_max"]), 1)

        #Calculate the two outputs using our simple rules above
        irrigation = get_irrigation(water)
        efficiency = get_efficiency(water, solar)

        #Add this row to our list
        rows.append({
            "temperature_c":       temp,
            "peak_solar_hours":    solar,
            "water_availability":  water,
            "soil_type":           soil,
            "humidity_pct":        humidity,
            "crop":                crop_info["crop"],
            "irrigation_timing":   irrigation,
            "resource_efficiency": efficiency,
        })

     #We add 20 extra rows per crop with slightly different conditions.
     #Keeps the same features the dataset grows bigger and the model sees overlaps.
     #Improves the Decision Tree / Random Forest accuracy
    for _ in range(50):
        temp_overlap = round(random.uniform(crop_info["temp_min"], crop_info["temp_max"]), 1)
        solar_overlap = round(random.uniform(crop_info["solar_min"], crop_info["solar_max"]), 1)
        water_overlap = random.choice(crop_info["water"])
        soil_overlap  = random.choice(crop_info["soils"])
        humidity_overlap = round(random.uniform(crop_info["humidity_min"], crop_info["humidity_max"]), 1)
    
        irrigation_overlap = get_irrigation(water_overlap)
        efficiency_overlap = get_efficiency(water_overlap, solar_overlap)

        rows.append({
        "temperature_c":       temp_overlap,
        "peak_solar_hours":    solar_overlap,
        "water_availability":  water_overlap,
        "soil_type":           soil_overlap,
        "humidity_pct":        humidity_overlap,
        "crop":                crop_info["crop"],
        "irrigation_timing":   irrigation_overlap,
        "resource_efficiency": efficiency_overlap,
    })

#Shuffle the rows so all the Wheat rows aren't bunched together,
#same for other crops.
random.shuffle(rows)

#4: Save to a CSV file
output_file = "sahrana_dataset.csv"
columns = ["temperature_c", "peak_solar_hours", "water_availability",
           "soil_type", "humidity_pct", "crop", "irrigation_timing", "resource_efficiency"]

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()    #method that writes the column names as the first row
    writer.writerows(rows)  #method that writes all the data rows

print(f"Done! Created {len(rows)} rows in {output_file}")
print(f"Crops: {len(CROP_CONDITIONS)}, rows per crop: {ROWS_PER_CROP}")
