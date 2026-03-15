import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

print("Loading cleaned dataset...")
# Load the data we cleaned in Step 2
df = pd.read_csv("cleaned_climate_data.csv")

# --- NEW FIX: Convert string coordinates to decimal floats ---
def parse_coord(coord):
    if isinstance(coord, str):
        val = float(coord[:-1]) # Get the numbers, ignore the last letter
        return -val if coord[-1] in ['S', 'W'] else val # Make S and W negative
    return coord

print("Parsing geographic coordinates...")
df['Latitude'] = df['Latitude'].apply(parse_coord)
df['Longitude'] = df['Longitude'].apply(parse_coord)
# -------------------------------------------------------------

# 1. Define Features and Targets
features = ['Year', 'Latitude', 'Longitude', 'Rainfall_mm', 'Humidity_pct', 'CO2_ppm', 'Seasonal_Variance']
X = df[features]

y_temp = df['Avg_Temp']          # Target for Regression
y_heatwave = df['Heatwave_Risk'] # Target for Classification

# 2. Train-Test Split (80% for training, 20% for testing)
X_train, X_test, y_temp_train, y_temp_test, y_hw_train, y_hw_test = train_test_split(
    X, y_temp, y_heatwave, test_size=0.2, random_state=42
)

# 3. Train Task 1: Temperature Prediction (Regression)
print("\nTraining Random Forest Regressor for Temperature Forecast...")
temp_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
temp_model.fit(X_train, y_temp_train)

# Evaluate the temperature model
temp_preds = temp_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_temp_test, temp_preds))
print(f"Temperature Model Root Mean Squared Error: {rmse:.2f}°C")

# 4. Train Task 2: Heatwave Risk (Classification)
print("\nTraining XGBoost Classifier for Heatwave Risk...")
hw_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
hw_model.fit(X_train, y_hw_train)

# Evaluate the heatwave model
hw_preds = hw_model.predict(X_test)
acc = accuracy_score(y_hw_test, hw_preds)
print(f"Heatwave Model Accuracy: {acc * 100:.2f}%")

# 5. Save the trained models to the saved_models folder
print("\nSaving models...")
os.makedirs("../saved_models", exist_ok=True)

with open("../saved_models/temperature_model.pkl", "wb") as f:
    pickle.dump(temp_model, f)

with open("../saved_models/heatwave_model.pkl", "wb") as f:
    pickle.dump(hw_model, f)

print("Success! Models saved to the 'saved_models' directory.")