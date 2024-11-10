import streamlit as st
import pandas as pd
import numpy as np
!pip install plotly
import plotly.graph_objs as go
from tensorflow.keras.models import load_model



model = load_model("lstmmodel.keras")


# Function to plot dataset
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

# Function to forecast future values
def forecast(model, last_input, future_steps=10):
    predictions = []
    current_input = last_input

    for _ in range(future_steps):
        pred = model.predict(current_input.reshape(1, 1, -1))
        predictions.append(pred[0, 0])

        # Update current input with the prediction to forecast further
        current_input = np.roll(current_input, -1)
        current_input[-1] = pred

    return np.array(predictions)

# Function to plot forecasted vs true values
def plot_forecast_interactive(true_values, future_predictions, forecast_steps=24):
    future_time_steps = np.arange(len(true_values), len(true_values) + forecast_steps)

    fig = go.Figure()

    # Plot true values
    fig.add_trace(go.Scatter(x=np.arange(len(true_values)), y=true_values, mode='lines', name='True Values', line=dict(color='blue')))
    
    # Plot forecasted values
    fig.add_trace(go.Scatter(x=future_time_steps, y=future_predictions, mode='lines', name='Forecasted Values', line=dict(color='red', dash='dash')))
    fig.update_layout(
        title="True vs. Forecasted Energy Consumption",
        xaxis_title="Time Steps",
        yaxis_title="Energy Consumption",
        template="plotly_dark",
        showlegend=True
    )

    st.plotly_chart(fig)

# Set background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_string}");
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Streamlit UI
st.title("Time Series Forecasting with LSTM")
set_background('https://drive.google.com/uc?id=15gmLQ9jUsYa_DFT9NxlbnJfDxMJXmlAU')  # Add the correct path to your background image

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Prepare the data
    if 'Datetime' not in df.index.names:
        df = df.set_index(['Datetime'])  # Set 'Datetime' as index if not already
    df = df.rename(columns={'PJME_MW': 'value'})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Plot dataset
    plot_dataset(df, title="Energy Consumption")

    # Select window size for forecasting
    window_size = st.slider("Select Window Size", min_value=10, max_value=200, value=60, step=10)

    # Prepare data for prediction
    X_test = df.value.values  # This is where you can preprocess the data as needed for your LSTM model

    # Get the last input for forecast
    last_input = X_test[-1]  # Or adjust as per your preprocessing
    future_predictions = forecast(model1, last_input, future_steps=window_size)

    # Plot the forecast
    plot_forecast_interactive(df.value.values[-window_size:], future_predictions, forecast_steps=window_size)

