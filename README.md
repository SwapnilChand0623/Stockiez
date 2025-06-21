# 📈 Stockiez – Stock Price Prediction API with LSTM 🤖

Welcome to **Stockiez**, a Flask-based web API that predicts stock prices using deep learning models! 🚀  
Powered by **LSTM neural networks**, this API delivers smart, time-series based forecasts for popular tickers like **SPY**, **AAPL**, **MSFT**, and **NKE**.

---
<img width="939" alt="image" src="https://github.com/user-attachments/assets/ca5f84f4-f058-453b-b0c9-2df9bdbfb78e" />


## 🔍 Main Features

- 📊 **Predictive Power**: Uses pre-trained LSTM models to forecast future stock prices  
- 🧠 **Machine Learning Integration**: Built with TensorFlow/Keras  
- 📀 **Historical Data**: Pulls stock data via Yahoo Finance (`yfinance`)  
- ⚙️ **Data Pipeline**:
  - Normalizes data with `MinMaxScaler`
  - Uses a 60-day rolling window (lookback) for model input
  - Returns clean actual vs predicted results

---

## 🛠️ API Endpoints

- `POST /predict` 📬  
  - **Input**: stock ticker, start date, end date  
  - **Output**: predicted vs actual prices + dates  

- `GET /health` ✅  
  - Check server status and available models

---

## 📊 Tech Stack & Logic Flow

1. 📟 Request received with ticker and date range  
2. 📅 Data fetched from Yahoo Finance  
3. 📉 Normalized and chunked into sequences  
4. 🔮 LSTM model makes predictions  
5. 📄 Denormalized results returned via JSON  

---
<img width="891" alt="image" src="https://github.com/user-attachments/assets/68afff18-7d07-4c0f-8051-b635d81f2e61" />


## 🧪 Error Handling & Configs

- Validates input dates, tickers, and model availability  
- Auto-selects available ports (starting at 5000)  
- Includes logging + CORS support for frontend integration 🌐

---

## 🌐 Use Case

Perfect for developers building stock prediction tools, dashboards, or mobile apps 📱.  
Just hook your frontend to Stockiez and start forecasting! 📉📈

---

> Production-ready with proper error handling, logging, model management, and web support.

Happy predicting! 🚀

