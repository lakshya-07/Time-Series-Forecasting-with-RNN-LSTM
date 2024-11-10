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
set_background('bg_tsa.jpg')  # Path to your image file


def create_lagged_features(data, n_lags=5):
    for lag in range(1, n_lags + 1):
        data[f'value_lag{lag}'] = data['value'].shift(lag)
    return data.dropna()


# Function to visualize the dataset
def plot_dataset(df, title):
    data = []

    value = go.Scatter(
        x=df.index,
        y=df.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    layout = dict(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    st.plotly_chart(fig)



# Forecasting function for model prediction
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
    """
    Plots the true values and forecasted future values interactively.

    Parameters:
        true_values (np.ndarray): Array of true values from the test set.
        future_predictions (np.ndarray): Array of forecasted values for the future.
        forecast_steps (int): Number of future time steps forecasted.
    """
    # Generate time steps for the forecasted period
    future_time_steps = np.arange(len(true_values), len(true_values) + forecast_steps)

    fig = go.Figure()

    # Change: Access the true_values array correctly, as it is 1-dimensional
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

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
model_choice = st.selectbox("Select Model", ("LSTM"))
window_size = st.slider("Select Window Size", min_value=10, max_value=200, value=60, step=10)

# Load model based on choice
model_dict = {
    "LSTM": "lstmmodel.keras"
}
model_path = model_dict[model_choice]
model = load_model(model_path)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'Datetime' not in df.index.names:
        df = df.set_index(['Datetime'])  # Set 'Datetime' as index if not already
    df = df.rename(columns={'PJME_MW': 'value'})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Plot the dataset
    plot_dataset(df, title='Energy Consumption (PJM East Region)')

    df = create_lagged_features(df)
    df = apply_ema(df)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['value'] + [f'value_lag{i}' for i in range(1, 6)] + ['value_ema']])
    X = scaled_data[:, 1:]  # Use lagged and EMA values as features
    y = scaled_data[:, 0]   # Original value column as target
    
    # Reshape for LSTM (samples, time steps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    last_input = X[-1, 0, :]  # Get last input for forecasting
    future_predictions = forecast(model, last_input, future_steps=window_size)

    # Plot the forecast
    plot_forecast_interactive(y[-window_size:], future_predictions, forecast_steps=window_size)
