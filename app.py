import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import joblib

st.set_page_config(page_title="Solar Irradiance Predictor")
st.title("\u2600\ufe0f Solar Irradiance Prediction App")

st.write("Enter the environmental conditions below to predict solar radiation:")

# Input features
Temperature = st.number_input("Temperature (°C)", value=25.0)
Pressure = st.number_input("Pressure (millibars)", value=1010.0)
Humidity = st.number_input("Humidity (%)", value=50.0)
WindDirection = st.number_input("Wind Direction (Degrees)", value=180.0)
Speed = st.number_input("Wind Speed (m/s)", value=3.0)

col1, col2, col3 = st.columns(3)
Month = col1.number_input("Month", min_value=1, max_value=12, value=6)
Day = col2.number_input("Day", min_value=1, max_value=31, value=15)
Hour = col3.number_input("Hour", min_value=0, max_value=23, value=12)

Minute = st.number_input("Minute", min_value=0, max_value=59, value=0)
Second = st.number_input("Second", min_value=0, max_value=59, value=0)

risehour = st.number_input("Sunrise Hour", min_value=0, max_value=6, value=6)
riseminuter = st.number_input("Sunrise Minute", min_value=0, max_value=59, value=0)
sethour = st.number_input("Sunset Hour", min_value=17, max_value=20, value=18)
setminute = st.number_input("Sunset Minute", min_value=0, max_value=59, value=30)

# Create input array
input_data = np.array([[
    Temperature, Pressure, Humidity, WindDirection, Speed,
    Month, Day, Hour, Minute, Second,
    risehour, riseminuter, sethour, setminute
]])

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("xgb_model.pkl")

# Scale input
scaled_input = scaler.transform(input_data)

if st.button("Predict Solar Radiation"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"\u2728 Predicted Solar Radiation: {prediction:.2f} W/m²")