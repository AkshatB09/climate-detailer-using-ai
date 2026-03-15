# 🌍 AI Climate Detailer

**An Interactive Global Climate Intelligence System**

[cite_start]Static climate dashboards are a thing of the past[cite: 62]. [cite_start]The AI Climate Detailer is a full-stack, AI-powered interactive global climate explorer built to provide new levels of interactivity, engagement, and foresight into global climate conditions[cite: 6, 68]. 

[cite_start]This project functions as an intermediate-to-advanced climate-tech startup prototype, bridging the gap between machine learning intelligence and intuitive frontend geospatial visualization [cite: 171, 189-195].

## 🚀 Core Features

* [cite_start]**3D Interactive World Map:** Explore the globe using a cinematic, 3D-tilted PyDeck interface that highlights global temperatures dynamically [cite: 90-94, 154-157].
* [cite_start]**Supervised Learning (AI Forecast):** Utilizes a **Random Forest Regressor** to predict future temperatures and an **XGBoost Classifier** to evaluate the risk of extreme heatwaves based on 10-20 years of historical data [cite: 104-122].
* [cite_start]**Vector Search (Contextual Intelligence):** Powered by a **FAISS vector database**, the system converts climate metrics into mathematical embeddings to answer complex contextual questions like, *"Which cities in history had similar climate conditions?"* [cite: 123-147].
* [cite_start]**Time Machine Slider:** Dynamically travel through time from historical periods to the present to watch climate metrics evolve [cite: 164-166].

## 🧠 System Architecture & Tech Stack

This platform relies on a streamlined, Python-native architecture:
* [cite_start]**Frontend UI:** Streamlit, PyDeck (3D Geospatial Mapping), Plotly Express [cite: 148-153].
* [cite_start]**Machine Learning Pipeline:** Scikit-Learn (Random Forest), XGBoost, Pandas, Numpy [cite: 116-118].
* [cite_start]**Vector Database:** FAISS (Facebook AI Similarity Search)[cite: 137, 187].

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
