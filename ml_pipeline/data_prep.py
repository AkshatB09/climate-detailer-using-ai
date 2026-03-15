import pandas as pd
import numpy as np

print("Loading dataset (this might take a moment)...")
# Load the raw data
df = pd.read_csv("GlobalLandTemperaturesByCity.csv")

# --- NEW FIX: Clean the coordinates at the source ---
def parse_coord(coord):
    if isinstance(coord, str):
        coord = coord.strip() # Remove any hidden spaces
        val = float(coord[:-1]) # Get the numbers, ignore the last letter
        return -val if coord[-1] in ['S', 'W'] else val # Make S and W negative
    return coord

print("Parsing geographic coordinates...")
df['Latitude'] = df['Latitude'].apply(parse_coord)
df['Longitude'] = df['Longitude'].apply(parse_coord)
# ---------------------------------------------------

# 1. Basic Cleaning & Date Extraction
print("Cleaning data...")
df = df.dropna(subset=['AverageTemperature']) # Remove rows with missing temperatures
df['dt'] = pd.to_datetime(df['dt'])
df['Year'] = df['dt'].dt.year
df['Month'] = df['dt'].dt.month

# Filter for recent history (e.g., year 2000 onwards)
df = df[df['Year'] >= 2000]

# 2. Group by City and Year to get Annual Averages
print("Calculating annual metrics...")
annual_data = df.groupby(['Country', 'City', 'Year', 'Latitude', 'Longitude']).agg(
    Avg_Temp=('AverageTemperature', 'mean'),
    Seasonal_Variance=('AverageTemperature', 'std') # Standard deviation represents seasonal variance
).reset_index()

# 3. Feature Engineering: Adding prototype features
# Since the base dataset lacks rainfall and humidity, we generate realistic synthetic 
# data based on the temperature to make the ML prototype functional.
np.random.seed(42)
annual_data['Rainfall_mm'] = np.clip(np.random.normal(800, 300, len(annual_data)) - (annual_data['Avg_Temp'] * 10), 50, 3000)
annual_data['Humidity_pct'] = np.clip(np.random.normal(60, 15, len(annual_data)) + (annual_data['Rainfall_mm'] * 0.01), 10, 100)
annual_data['CO2_ppm'] = 370 + (annual_data['Year'] - 2000) * 2.1 # Approximating global CO2 growth

# Create a mock target variable for Heatwave Risk Classification (1 = Yes, 0 = No)
# Let's say a heatwave is likely if the temp is abnormally high for that city
annual_data['Heatwave_Risk'] = (annual_data['Avg_Temp'] > annual_data.groupby('City')['Avg_Temp'].transform('mean') + 1.5).astype(int)

# 4. Save the cleaned dataset
output_path = "cleaned_climate_data.csv"
annual_data.to_csv(output_path, index=False)
print(f"Success! Cleaned data saved to {output_path}")
print(annual_data.head())