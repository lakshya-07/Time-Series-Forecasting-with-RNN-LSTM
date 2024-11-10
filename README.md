# **Time Series Forecasting with LSTM**

This repository demonstrates how to perform time series forecasting using Long Short-Term Memory (LSTM) networks. The project includes the code for data preprocessing, building, training, and evaluating LSTM models, as well as interactive visualizations for both the time series data and forecasted results. 

## **Introduction**

Time series forecasting is a method used to predict future values based on previously observed values. This repository specifically focuses on forecasting using LSTM (Long Short-Term Memory) networks, a type of recurrent neural network (RNN) known for its effectiveness in sequential data modeling.

The goal of this project is to predict future energy consumption values based on historical data. The repository provides code for preprocessing data, applying the LSTM model, and evaluating its performance.

---
## **Dataset**
The dataset used in this repository consists of hourly energy consumption data, with two columns:

Datetime: Timestamp of the energy consumption data point.
Value: Energy consumption in megawatts (MW).
Ensure your dataset is in CSV format before uploading to the app.

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
