# 🌍 AI Climate Detailer

**An Interactive Global Climate Intelligence System**

Static climate dashboards are a thing of the past. The AI Climate Detailer is a full-stack, AI-powered interactive global climate explorer built to provide new levels of interactivity, engagement, and foresight into global climate conditions. 

This project functions as a climate-tech startup prototype, bridging the gap between machine learning intelligence and intuitive frontend geospatial visualization.

## 🚀 Core Features

* **3D Interactive World Map:** Explore the globe using a cinematic, 3D-tilted PyDeck interface.
* **Supervised Learning (AI Forecast):** Utilizes a **Random Forest Regressor** to predict future temperatures and an **XGBoost Classifier** to evaluate the risk of extreme heatwaves.
* **Vector Search (Contextual Intelligence):** Powered by a **FAISS vector database**, the system converts climate metrics into mathematical embeddings to answer complex contextual questions like, *"Which cities in history had similar climate conditions?"*
* **Time Machine Slider:** Dynamically travel through time to watch historical temperature trends, rainfall, and climate metrics evolve over the decades.

## 🧠 System Architecture & Tech Stack

This platform relies on a streamlined, Python-native architecture:
* **Frontend UI:** Streamlit, PyDeck (3D Geospatial Mapping), Plotly Express
* **Machine Learning Pipeline:** Scikit-Learn (Random Forest), XGBoost, Pandas, Numpy
* **Vector Database:** FAISS (Facebook AI Similarity Search)

## 📂 Project Structure

```text
ai-climate-detailer/
├── ml_pipeline/                 # Data preprocessing, training, and vectorization scripts
│   ├── data_prep.py             # Cleans raw data and engineers seasonal/climate features
│   ├── train_models.py          # Trains RF and XGBoost models
│   └── build_vector_db.py       # Normalizes data and builds the FAISS index
├── saved_models/                # Stores compiled .pkl models, the FAISS index, and metadata
├── app.py                       # The main Streamlit application and UI
└── requirements.txt             # Python dependencies