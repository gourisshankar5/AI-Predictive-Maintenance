import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import IsolationForest

# Function to generate one row of sensor data
def generate_sensor_row():
    timestamp = pd.Timestamp.now()
    temperature = np.random.normal(loc=75, scale=5)   # °C
    vibration = np.random.normal(loc=2, scale=0.5)    # mm/s
    current = np.random.normal(loc=15, scale=3)       # A
    power = current * np.random.normal(loc=0.75, scale=0.05)  # kW
    return [timestamp, temperature, vibration, current, power]

# Streamlit page setup
st.set_page_config(page_title="AI-Powered Predictive Maintenance", layout="wide")
st.title("⚡ Real-Time Sensor Data with AI Anomaly Detection")

# Initialize empty DataFrame
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(
        columns=["timestamp", "temperature_C", "vibration_mm_s", "current_A", "power_kW"]
    )
    st.session_state.model = IsolationForest(contamination=0.1, random_state=42)  # AI model
    st.session_state.model_trained = False

# Placeholders
chart_placeholder = st.empty()
alert_placeholder = st.empty()

# Generate data continuously
while True:
    new_row = generate_sensor_row()
    new_df = pd.DataFrame([new_row], columns=st.session_state.data.columns)

    # Append new row
    st.session_state.data = pd.concat([st.session_state.data, new_df], ignore_index=True)

    # Keep only last 200 rows for training/display
    display_df = st.session_state.data.tail(200)

    # Train the model once we have enough data
    if len(display_df) > 30:
        X = display_df[["temperature_C", "vibration_mm_s", "current_A", "power_kW"]]
        st.session_state.model.fit(X)
        st.session_state.model_trained = True

    # Show live charts
    with chart_placeholder.container():
        st.line_chart(display_df.set_index("timestamp")[["temperature_C", "vibration_mm_s"]])
        st.line_chart(display_df.set_index("timestamp")[["current_A", "power_kW"]])
        st.dataframe(display_df.tail(5))  # latest values

    # AI anomaly detection
    if st.session_state.model_trained:
        latest = display_df.iloc[-1][["temperature_C", "vibration_mm_s", "current_A", "power_kW"]].values.reshape(1, -1)
        prediction = st.session_state.model.predict(latest)[0]  # 1 = normal, -1 = anomaly

        with alert_placeholder.container():
            st.subheader("🔍 AI Predictive Maintenance Alerts")
            if prediction == -1:
                st.error("⚠️ AI Model: Anomaly detected! Possible failure risk.")
            else:
                st.success("✅ AI Model: Machine operating normally.")

    time.sleep(1)  # refresh every second