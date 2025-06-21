# train_models.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# List of tickers to train
TICKERS = ['SPY', 'AAPL', 'MSFT', 'NKE']
START_DATE = '2010-01-01'
END_DATE = '2023-01-01'
LOOK_BACK = 60

def create_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LOOK_BACK, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_save(ticker):
    print(f"\nTraining model for {ticker}...")
    
    # Download data
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    close_prices = data['Close'].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Create sequences
    def create_sequences(dataset):
        X, y = [], []
        for i in range(LOOK_BACK, len(dataset)):
            X.append(dataset[i-LOOK_BACK:i, 0])
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(scaled_data)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Train model
    model = create_model()
    model.fit(X, y, epochs=30, batch_size=32, verbose=1)
    
    # Save model
    model_path = f'models/{ticker}_lstm_model.h5'
    model.save(model_path)
    print(f"Saved {ticker} model to {model_path}")

if __name__ == '__main__':
    for ticker in TICKERS:
        train_and_save(ticker)
    print("\nAll models trained and saved successfully!")