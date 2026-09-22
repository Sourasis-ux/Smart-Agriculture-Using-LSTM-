# 🌱 Smart Agriculture IoT Alert System

## Overview
This is an end-to-end predictive maintenance and automated alert pipeline for smart agriculture. It combines Deep Learning for time-series forecasting with a local Large Language Model (LLM) to translate raw mathematical predictions into actionable, human-readable insights for farm managers.

## Tech Stack
*   **Deep Learning:** TensorFlow, Keras (LSTM)
*   **Generative AI:** Ollama, Llama 3.2
*   **Data Processing:** Pandas, NumPy, Scikit-Learn (MinMaxScaler)
*   **Frontend/Deployment:** Streamlit

## Project Architecture
The pipeline is modularized into four distinct stages:

1.  **`data_prep.py` (Data Ingestion & Prep):**
    *   Loads historical IoT sensor data (`SmartFarm.csv`).
    *   Cleans missing data using forward fill (`.ffill()`).
    *   Scales features between [0, 1] using `MinMaxScaler`.
    *   Creates 24-hour sliding windows (sequences) to provide temporal context for the neural network.

2.  **`train_lstm.py` (Predictive Modeling):**
    *   Performs a strict chronological Train/Test split (`shuffle=False`) to prevent time-series data leakage.
    *   Trains a Recurrent Neural Network using an `LSTM` layer (50 units) and a `Dense` output layer (3 units).
    *   Optimizes using Adam and Mean Squared Error (MSE), saving the final weights to `farm_lstm.keras`.

3.  **`llm_agent.py` (Generative AI Bridge):**
    *   Acts as the "Incident Commander."
    *   Takes the raw float predictions (Moisture, Temperature, Humidity) and dynamically injects them into an engineered prompt.
    *   Sends a local, zero-latency POST request to Ollama (Llama 3.2) to generate a brief diagnostic report.

4.  **`dashboard.py` (Production UI):**
    *   An interactive Streamlit application.
    *   Caches the deep learning model into memory to prevent UI freezing.
    *   Performs inverse algebraic transformations on the LSTM's output to display real-world metrics.
    *   Triggers the forecasting and LLM generation pipelines simultaneously.

## How to Run
1. Ensure you have your Python environment activated with all dependencies installed (`tensorflow`, `streamlit`, `pandas`, `scikit-learn`, `requests`).
2. Make sure your local **Ollama** server is running in the background with the Llama 3.2 model downloaded (`ollama run llama3.2`).
3. Execute the dashboard from your terminal:
   ```bash
   streamlit run dashboard.py
   ```