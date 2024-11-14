# **Time Series Forecasting with LSTM**

This repository demonstrates how to perform time series forecasting using Simple Recurrent Neural Network (Simple RNNS) and Long Short-Term Memory (LSTM) networks. The project includes the code for data preprocessing, building, training, and evaluating LSTM models, as well as interactive visualizations for both the time series data and forecasted results. 
The goal of this project is to predict future energy consumption values based on historical data. The repository provides code for preprocessing data, applying the LSTM model, and evaluating its performance.

## **Overview of the Time Series Forecasting App**

 APP LINK: https://time-series-forecasting-with-rnn-lstm-kfghpmsicgv6fiexmrp9tj.streamlit.app/
 
Welcome to the Time Series Forecasting App! This app provides an interactive platform for time
 series analysis and forecasting using deep learning models, specifically focused on LSTM (Long Short-Term
 Memory) networks. Built with a user-friendly interface, this app is tailored to predict future values based
 energy consumption trends. The model is designed to capture temporal patterns in the data to make reliable
 future forecasts

  ## **How to Use the App**
 1. **Upload Your Dataset:**
 • Ensure your file is a CSV format with a Datetime column (for dates) and a value column (the target
 variable you wish to forecast, such as energy consumption or stock prices).
 • Upload the file directly in the app, which will automatically parse and prepare it for analysis.
 2. **Configure Forecasting Settings:**
 • Choose a model (LSTM) for forecasting. This model is pre-loaded and ready to process your data.
 • Adjust the window size slider to define the desired forecasting horizon. The window size controls
 the number of future time steps the model will predict based on the recent past.
 3. **Generate and Visualize Forecast:**
 • After uploading and configuring, the app will preprocess the data, apply transformations (such as
 lagging and exponential moving average), and generate forecasts using the trained LSTM model.
 • Viewthe interactive visualization that compares actual data (in blue) with forecasted values (in red).
 This plot offers insights into model accuracy and forecast trends.
 
## **Dataset**
The dataset used in this repository consists of hourly energy consumption data, with two columns:

 • Datetime: Timestamp of the energy consumption data point.
 • Value: Energy consumption in megawatts (MW).
Ensure your dataset is in CSV format before uploading to the app.


## Code Details
1. **Data Preprocessing**: Includes lagged features and exponential moving averages.
2. **Model Training**: Utilizes Simple RNN and LSTM for forecasting.
3. **Evaluation**: Model performance evaluated using MAE, MSE, RMSE, and R².

---

## **Visualization**
This project includes two types of visualizations:

Time Series Plot: Plots the historical data, allowing you to visually inspect trends and patterns.
Forecast Plot: Compares the true values of the test set with the forecasted values from the LSTM model. You can interactively view how well the model predicts future values.

## **Evaluation Metrics**
The following metrics are used to evaluate the performance of the LSTM model:

Mean Absolute Error (MAE): Measures the average magnitude of the errors in a set of predictions, without considering their direction.
Mean Squared Error (MSE): Measures the average of the squared differences between the predicted and actual values.
Root Mean Squared Error (RMSE): The square root of MSE, providing an error measure in the same units as the original data.
Mean Absolute Percentage Error (MAPE): Measures the accuracy of the forecast as a percentage.

### **Dependencies**

To run this project, you'll need the following Python packages:

- `numpy`
- `pandas`
- `tensorflow`
- `plotly`
- `streamlit`
- `scikit-learn`
- `matplotlib`

You can install these packages using `pip`:

```bash
pip install numpy pandas tensorflow plotly streamlit scikit-learn matplotlib
