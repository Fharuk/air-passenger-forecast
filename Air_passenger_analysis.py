# -----------------------------
# streamlit_app_final.py
# -----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import matplotlib.pyplot as plt
from io import BytesIO

# -----------------------------
# Load saved models
# -----------------------------
sarima_model = SARIMAXResults.load('sarima_model.pkl')
lstm_model = load_model('lstm_residual_model_pred.h5')  # prediction-only
residual_scaler = joblib.load('residual_scaler.pkl')

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv('AirPassengers.csv')
df['Log_Passengers'] = np.log(df['#Passengers'])

# -----------------------------
# Streamlit Sidebar
# -----------------------------
st.sidebar.header("Forecast Settings")
forecast_months = st.sidebar.slider("Number of months to forecast", 1, 36, 12)

st.sidebar.header("Add Recent Passenger Data")
new_passengers_input = st.sidebar.text_area(
    "Enter new monthly passenger counts separated by commas (optional)",
    ""
)

# Append new data if provided
if new_passengers_input.strip():
    try:
        new_passengers = [float(x.strip()) for x in new_passengers_input.split(',')]
        last_month = pd.to_datetime(df['Month'].iloc[-1])
        new_dates = pd.date_range(start=last_month + pd.offsets.MonthBegin(1), periods=len(new_passengers), freq='MS')
        new_df = pd.DataFrame({'Month': new_dates, '#Passengers': new_passengers})
        new_df['Log_Passengers'] = np.log(new_df['#Passengers'])
        df = pd.concat([df, new_df], ignore_index=True)
        st.success(f"Added {len(new_passengers)} new months of data.")
    except:
        st.error("Invalid input format. Use comma-separated numbers.")

# -----------------------------
# App Title
# -----------------------------
st.title("AirPassengers Hybrid Forecast (Prediction-Only LSTM)")
st.write("Hybrid SARIMA + LSTM forecasting with confidence intervals and CSV download.")

# -----------------------------
# Display Historical Data
# -----------------------------
st.subheader("Historical Data (Last 10 points)")
st.dataframe(df.tail(10))

# -----------------------------
# SARIMA Forecast
# -----------------------------
forecast_steps = forecast_months
sarima_forecast = sarima_model.get_forecast(steps=forecast_steps)
sarima_forecast_mean = sarima_forecast.predicted_mean.values
sarima_ci = sarima_forecast.conf_int(alpha=0.05)  # 95% confidence interval
sarima_ci_lower = np.exp(sarima_ci.iloc[:,0])
sarima_ci_upper = np.exp(sarima_ci.iloc[:,1])

# -----------------------------
# LSTM Residual Forecast
# -----------------------------
seq_length = 12
residuals = df['Log_Passengers'].values[-seq_length:].reshape(-1,1)
residuals_scaled = residual_scaler.transform(residuals)

lstm_pred_scaled = []
last_seq = residuals_scaled.copy()

for i in range(forecast_steps):
    pred_scaled = lstm_model.predict(last_seq.reshape(1, seq_length, 1), verbose=0)[0,0]
    lstm_pred_scaled.append(pred_scaled)
    last_seq = np.vstack([last_seq[1:], [[pred_scaled]]])

lstm_pred = residual_scaler.inverse_transform(np.array(lstm_pred_scaled).reshape(-1,1)).flatten()

# -----------------------------
# Hybrid Forecast
# -----------------------------
hybrid_forecast_log = sarima_forecast_mean + lstm_pred
hybrid_forecast = np.exp(hybrid_forecast_log)

# Forecast dates
last_date = pd.to_datetime(df['Month'].iloc[-1])
future_months = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=forecast_months, freq='MS')
forecast_df = pd.DataFrame({
    'Month': future_months,
    'Forecasted_Passengers': hybrid_forecast,
    'CI_Lower': sarima_ci_lower[:forecast_steps],
    'CI_Upper': sarima_ci_upper[:forecast_steps]
})

# -----------------------------
# Display Forecast Table
# -----------------------------
st.subheader(f"Next {forecast_months} Months Forecast with 95% CI")
st.dataframe(forecast_df)

# Download CSV
def convert_df_to_csv(df):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer

csv_data = convert_df_to_csv(forecast_df)
st.download_button("Download Forecast as CSV", csv_data, file_name="hybrid_forecast.csv", mime="text/csv")

# -----------------------------
# Plot Historical + Forecast with CI
# -----------------------------
st.subheader("Historical + Forecast Plot with 95% CI")
plt.figure(figsize=(12,6))
plt.plot(pd.to_datetime(df['Month']), df['#Passengers'], label='Historical')
plt.plot(future_months, hybrid_forecast, label='Hybrid Forecast', color='green')
plt.fill_between(future_months, forecast_df['CI_Lower'], forecast_df['CI_Upper'], color='green', alpha=0.2, label='95% CI')
plt.xlabel('Month')
plt.ylabel('Number of Passengers')
plt.title('AirPassengers: Historical + Hybrid Forecast with 95% CI')
plt.legend()
plt.grid(True)
st.pyplot(plt)
