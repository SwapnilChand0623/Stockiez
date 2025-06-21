# Stock Market Prediction with Multiple Tickers - Final Corrected Version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf

# ======================
# 1. Download Multiple Stock Data - PROPERLY HANDLED
# ======================
tickers = ['^GSPC', 'NKE', 'MSFT', 'SBUX']  # S&P 500, Nike, Microsoft, Starbucks
start_date = '2010-01-01'
end_date = '2023-01-01'

print("Downloading stock data for multiple tickers...")
close_prices = pd.DataFrame()

for ticker in tickers:
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # PROPER way to add to DataFrame - no to_frame() needed
        close_prices[ticker] = data['Close']
        print(f"Successfully downloaded {ticker} data")
    except Exception as e:
        print(f"Failed to download {ticker}: {str(e)}")
        continue

if close_prices.empty:
    raise ValueError("No data was downloaded successfully - check your ticker symbols")

# Save the data
close_prices.to_csv('multiple_stocks_data.csv')
print("\nData Overview:")
print(close_prices.head())

# ======================
# 2. Data Preparation
# ======================
# Normalize each stock separately
scalers = {}
scaled_data = pd.DataFrame()

for ticker in close_prices.columns:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(close_prices[[ticker]].values)
    scaled_data[ticker] = scaled_values.flatten()
    scalers[ticker] = scaler

# ======================
# 3. Create Training Data for Each Ticker
# ======================
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i])
        y.append(dataset[i])
    return np.array(X), np.array(y)

look_back = 60
models = {}
predictions_dict = {}
actual_values_dict = {}

for ticker in close_prices.columns:
    print(f"\nProcessing {ticker}...")
    try:
        # Prepare data
        dataset = scaled_data[ticker].values
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_data = dataset[:train_size]
        test_data = dataset[train_size:]
        
        # Create sequences
        X_train, y_train = create_dataset(train_data, look_back)
        X_test, y_test = create_dataset(test_data, look_back)
        
        # Reshape for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        print(f"Training {ticker} model...")
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
        
        # Make predictions
        test_predictions = model.predict(X_test)
        test_predictions = scalers[ticker].inverse_transform(test_predictions)
        actual_values = scalers[ticker].inverse_transform(y_test.reshape(-1, 1))
        
        predictions_dict[ticker] = test_predictions
        actual_values_dict[ticker] = actual_values
        models[ticker] = model
        print(f"Completed {ticker} successfully")
        
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        continue

# ======================
# 4. Visualize Results
# ======================
if predictions_dict:
    plt.figure(figsize=(16, 10))
    
    for i, ticker in enumerate(predictions_dict.keys()):
        plt.subplot(2, 2, i+1)
        plt.plot(actual_values_dict[ticker], label='Actual', color='blue')
        plt.plot(predictions_dict[ticker], label='Predicted', color='red', alpha=0.7)
        plt.title(f'{ticker} Stock Prediction')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('multiple_stocks_prediction.png', dpi=300)
    plt.show()
    
    # Save models
    for ticker, model in models.items():
        model.save(f'{ticker}_lstm_model.h5')
        print(f"Saved model for {ticker}")
else:
    print("No successful predictions to display")

print("\nScript completed!")