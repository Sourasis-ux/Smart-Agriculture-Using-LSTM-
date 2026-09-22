import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from llm_agent import generate_agricultural_report

st.set_page_config(page_title="Smart Farm AI", page_icon="🌱", layout="wide")
st.title("🌱 IoT Predictive Maintenance & Alert System")
st.markdown("Powered by TensorFlow (LSTM) & Llama 3.2")

@st.cache_resource
def load_system():
    
    model = load_model("farm_lstm.keras")
    
    
    df = pd.read_csv("SmartFarm.csv").ffill()
    features = ['soil_moisture', 'temperature', 'humidity']
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    return model, df, scaler, features, scaled_data

try:
    model, df, scaler, features, scaled_data = load_system()
    
    st.subheader("Live Sensor History (Last 24 Hours)")
    recent_data = df[features].tail(24)
    st.line_chart(recent_data)
    
    if st.button("🔮 Forecast Next Hour & Generate AI Report", type="primary"):
        with st.spinner("LSTM predicting future state..."):
            # 1. Grab the last 24 rows as our input window
            last_24 = scaled_data[-24:]
            
            
            X_input = last_24.reshape(1, 24, 3)
            
            
            pred_scaled = model.predict(X_input)
            
            
            pred_real = scaler.inverse_transform(pred_scaled)[0]
            moisture, temp, humidity = pred_real
            
        
        st.success("Deep Learning Forecast Complete")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Moisture", f"{moisture:.2f}")
        col2.metric("Predicted Temp", f"{temp:.2f} °C")
        col3.metric("Predicted Humidity", f"{humidity:.2f} %")
        
        
        with st.spinner("Llama 3.2 analyzing forecast..."):
            report = generate_agricultural_report(moisture, temp, humidity)
            
        st.warning("### 🤖 AI Incident Commander Report")
        st.write(report)
            
except Exception as e:
    st.error(f"System Error: Make sure 'farm_lstm.keras' and 'SmartFarm.csv' are in the same folder. Details: {e}")