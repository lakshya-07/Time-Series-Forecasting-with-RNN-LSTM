import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import base64

# Set up the background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_string}");
                background-size: cover;
                color: white; /* Set font color to white */
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Apply the background
set_background('bg_tsa.jpg')

# Function to create lagged features
def create_lagged_features(data, n_lags=5):
    for lag in range(1, n_lags + 1):
        data[f'value_lag{lag}'] = data['value'].shift(lag)
    return data.dropna()

# Function to apply EMA
def apply_ema(data, span=10):
    data['value_ema'] = data['value'].ewm(span=span, adjust=False).mean()
    return data

# Function to visualize the dataset
def plot_dataset(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df.value, mode="lines", name="values", line=dict(color="rgba(0,0,0,0.3)")))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False)
    )
    st.plotly_chart(fig)

# Forecasting function
def forecast(model, last_input, future_steps=24):
    predictions = []
    current_input = last_input

    for _ in range(future_steps):
        pred = model.predict(current_input.reshape(1, 1, -1))
        predictions.append(pred[0, 0])
        current_input = np.roll(current_input, -1)
        current_input[-1] = pred

    return np.array(predictions)

# Function to visualize forecast results interactively
def plot_forecast_interactive(true_values, future_predictions, forecast_steps=24):
    future_time_steps = np.arange(len(true_values), len(true_values) + forecast_steps)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(true_values)), y=true_values, mode='lines', name='True Values', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_time_steps, y=future_predictions, mode='lines', name='Forecasted Values', line=dict(color='red', dash='dash')))
    fig.update_layout(
        title="True and Forecasted Energy Consumption",
        xaxis_title="Time Steps",
        yaxis_title="Energy Consumption",
        template="plotly_dark",
        showlegend=True
    )
    st.plotly_chart(fig)

# Streamlit App UI
st.title("Time Series Forecasting App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
    if 'PJME_MW' in df.columns:
        df = df.rename(columns={'PJME_MW': 'value'})
    
    df = df.sort_index()
    plot_dataset(df, title='Energy Consumption (PJM East Region)')

    # Preprocess data
    df = create_lagged_features(df)
    df = apply_ema(df)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['value'] + [f'value_lag{i}' for i in range(1, 6)] + ['value_ema']])
    X = scaled_data[:, 1:]  # Use lagged and EMA values as features
    y = scaled_data[:, 0]   # Original value column as target
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM (samples, time steps, features)

    # Model selection and forecast
    model_choice = st.selectbox("Select Model", ("LSTM"))
    model_path = "lstmmodel.keras"  # Assuming "lstmmodel.keras" is available
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")

    window_size = st.slider("Select Window Size", min_value=10, max_value=200, value=60, step=10)
    last_input = X[-1, 0, :]  # Get last input for forecasting
    future_predictions = forecast(model, last_input, future_steps=window_size)

    # Plot the forecast
    plot_forecast_interactive(y[-window_size:], future_predictions, forecast_steps=window_size)
