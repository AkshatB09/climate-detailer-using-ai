import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sklearn.preprocessing import StandardScaler

print("Loading cleaned dataset for Vectorization...")
df = pd.read_csv("cleaned_climate_data.csv")

# 1. Select the features to build the vector
# According to the project blueprint, the vector includes:
# Avg temperature, Rainfall, Humidity, CO2 level, Latitude, Seasonal variance
vector_features = ['Avg_Temp', 'Rainfall_mm', 'Humidity_pct', 'CO2_ppm', 'Latitude', 'Seasonal_Variance']
X_vectors = df[vector_features]

# 2. Normalize the Data
# FAISS calculates distances between points. If CO2 is in the 400s and Temp is 30, 
# the math gets skewed. StandardScaler puts everything on the same scale.
print("Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_vectors)

# FAISS strictly requires float32 data types
X_scaled = np.array(X_scaled, dtype=np.float32)

# 3. Build the FAISS Index
print("Building FAISS index...")
dimension = X_scaled.shape[1] # The number of features in our vector (6)
index = faiss.IndexFlatL2(dimension) # L2 distance (Euclidean) for similarity search

# Add our climate vectors to the index
index.add(X_scaled)
print(f"Total vectors added to FAISS: {index.ntotal}")

# 4. Save the Index, Scaler, and Metadata
print("Saving vector database and metadata...")
os.makedirs("../saved_models", exist_ok=True)

# Save the FAISS index
faiss.write_index(index, "../saved_models/climate_vectors.index")

# Save the scaler (we need this in the app to scale the user's selected city before searching)
with open("../saved_models/vector_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save the metadata (Country, City, Year) so when FAISS returns an ID (like 1042), 
# we know which city that ID actually belongs to!
metadata = df[['Country', 'City', 'Year', 'Avg_Temp']]
metadata.to_csv("C:\\Users\\aksha\\Downloads\\ai-climate-detailer\\saved_models\\vector_metadata.csv", index=False)

print("Success! FAISS Index and metadata saved to 'saved_models'.")