import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
import plotly.express as px
import pydeck as pdk

# 1. Page Configuration (Modern Dark Theme)
st.set_page_config(page_title="AI Climate Detailer", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #1E90FF; font-weight: 700; }
    .stMetric { background-color: #1E1E1E; padding: 15px; border-radius: 10px; border-left: 4px solid #1E90FF; }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 AI Climate Detailer")
st.markdown("### Interactive Global Climate Intelligence System")

# 2. Load Data and Models
@st.cache_data
def load_data():
    return pd.read_csv("ml_pipeline/cleaned_climate_data.csv")

@st.cache_resource
def load_models():
    with open("saved_models/temperature_model.pkl", "rb") as f:
        temp_model = pickle.load(f)
    with open("saved_models/heatwave_model.pkl", "rb") as f:
        hw_model = pickle.load(f)
    with open("saved_models/vector_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    index = faiss.read_index("saved_models/climate_vectors.index")
    metadata = pd.read_pickle("saved_models/vector_metadata.pkl")
    return temp_model, hw_model, scaler, index, metadata

df = load_data()
temp_model, hw_model, scaler, index, metadata = load_models()

# 3. Interactive UI: City & Time Selection
st.sidebar.title("Navigation")
cities = sorted(df['City'].unique())
selected_city = st.sidebar.selectbox("Search for a City:", cities)

if selected_city:
    # Isolate the history for the chosen city
    city_history = df[df['City'] == selected_city].sort_values(by='Year')
    
    # --- NEW: Time Slider ---
    min_year = int(city_history['Year'].min())
    max_year = int(city_history['Year'].max())
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕰️ Time Machine")
    selected_year = st.sidebar.slider("Select a year to explore:", min_value=min_year, max_value=max_year, value=max_year, step=1)
    
    # Filter the exact data for the year the user just selected on the slider
    city_data = city_history[city_history['Year'] == selected_year].iloc[0]
    
    col1, col2 = st.columns([1.5, 1])
    
    # --- PANEL 1: 3D Cinematic Map (PyDeck) ---
    with col1:
        st.subheader(f"📍 Global View: {selected_city}, {city_data['Country']} ({selected_year})")
        
        map_data = pd.DataFrame({'City': [selected_city], 'Lat': [city_data['Latitude']], 'Lon': [city_data['Longitude']]})
        
        view_state = pdk.ViewState(
            latitude=city_data['Latitude'], 
            longitude=city_data['Longitude'], 
            zoom=5, 
            pitch=45 
        )
        
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=map_data,
            get_position='[Lon, Lat]',
            get_color='[255, 75, 75, 200]',
            get_radius=30000,
            pickable=True
        )
        
        st.pydeck_chart(pdk.Deck(
            layers=[layer], 
            initial_view_state=view_state, 
            map_style='dark',
            tooltip={"text": "{City}"}
        ))
        
        st.subheader("📈 Historical Temperature Trend")
        fig = px.line(city_history, x='Year', y='Avg_Temp', markers=True)
        # Highlight the selected year on the chart
        fig.add_vline(x=selected_year, line_width=2, line_dash="dash", line_color="red")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
        fig.update_traces(line_color='#1E90FF', line_width=3)
        st.plotly_chart(fig, use_container_width=True)

    # --- PANEL 2: AI Forecast & Metrics ---
    with col2:
        st.subheader(f"📊 Intelligence for {selected_year}")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Avg Temperature", f"{city_data['Avg_Temp']:.1f}°C")
        metric_col2.metric("Rainfall", f"{city_data['Rainfall_mm']:.0f} mm")
        
        st.divider()
        forecast_year = selected_year + 5
        st.subheader(f"🤖 AI Forecast ({forecast_year})")
        
        future_features = pd.DataFrame({
            'Year': [forecast_year],
            'Latitude': [city_data['Latitude']],
            'Longitude': [city_data['Longitude']],
            'Rainfall_mm': [city_data['Rainfall_mm']],
            'Humidity_pct': [city_data['Humidity_pct']],
            'CO2_ppm': [city_data['CO2_ppm'] + 10], 
            'Seasonal_Variance': [city_data['Seasonal_Variance']]
        })
        
        pred_temp = temp_model.predict(future_features)[0]
        pred_hw = hw_model.predict(future_features)[0]
        
        st.metric("Predicted Temp", f"{pred_temp:.1f}°C", delta=f"{pred_temp - city_data['Avg_Temp']:.1f}°C")
        if pred_hw == 1:
            st.error("⚠️ Heatwave Risk: HIGH DANGER")
        else:
            st.success("🟢 Heatwave Risk: NORMAL")

        st.divider()
        
        # --- PANEL 3: Vector Search ---
        st.subheader("🔍 Contextual Intelligence")
        st.write(f"Cities in history similar to {selected_city} in {selected_year}:")
        
        vector_features = ['Avg_Temp', 'Rainfall_mm', 'Humidity_pct', 'CO2_ppm', 'Latitude', 'Seasonal_Variance']
        city_vector = city_data[vector_features].values.reshape(1, -1)
        scaled_vector = scaler.transform(city_vector).astype(np.float32)
        
        distances, indices = index.search(scaled_vector, 4) 
        
        for i in range(1, 4): 
            match_idx = indices[0][i]
            match_meta = metadata.iloc[match_idx]
            st.info(
                f"**{match_meta['City']}, {match_meta['Country']} ({match_meta['Year']})**\n\n"
                f"Avg Temp: {match_meta['Avg_Temp']:.1f}°C"
            )

'''   
    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
    venv\Scripts\activate
    python -m streamlit run app.py
'''