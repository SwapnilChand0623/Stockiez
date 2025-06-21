import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pandas as pd
from datetime import datetime, timedelta
import socket
from contextlib import closing
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODELS_DIR = 'models'
SUPPORTED_TICKERS = ['SPY', 'AAPL', 'MSFT', 'NKE']
LOOK_BACK = 60
DEFAULT_PORT = 5000
MAX_PORT_ATTEMPTS = 10

# Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def load_all_models():
    """Load all available models with error handling"""
    models = {}
    for ticker in SUPPORTED_TICKERS:
        model_path = os.path.join(MODELS_DIR, f'{ticker}_lstm_model.h5')
        try:
            if os.path.exists(model_path):
                # Disable Keras warnings during loading
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    models[ticker] = load_model(model_path)
                logger.info(f"Successfully loaded model for {ticker}")
            else:
                logger.warning(f"Model not found for {ticker}")
        except Exception as e:
            logger.error(f"Failed to load model for {ticker}: {str(e)}")
    return models

models = load_all_models()

def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

def predict_future_prices(model, last_sequence, scaler, future_days=30):
    """
    Predict future prices using the LSTM model with recursive prediction
    Args:
        model: Loaded LSTM model
        last_sequence: Last sequence of historical data (scaled)
        scaler: The scaler used for normalization
        future_days: Number of days to predict ahead
    Returns:
        Array of predicted prices (unscaled)
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(future_days):
        # Predict next day
        next_pred = model.predict(current_sequence.reshape(1, LOOK_BACK, 1), verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Update sequence: drop oldest, add new prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred[0, 0]
    
    # Inverse transform the predictions
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        future_days = int(data.get('future_days', 0))  # New parameter for future prediction days
        
        # Validate input
        if not all([ticker, start_date, end_date]):
            return jsonify({'error': 'Missing required parameters'}), 400
            
        if ticker not in SUPPORTED_TICKERS:
            return jsonify({'error': f'Unsupported ticker. Supported: {", ".join(SUPPORTED_TICKERS)}'}), 400
            
        if ticker not in models:
            return jsonify({'error': f'Model for {ticker} not available'}), 503
            
        # Date validation
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            if start_dt >= end_dt:
                return jsonify({'error': 'Start date must be before end date'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
            
        # Download data
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            return jsonify({'error': 'No data available for the specified date range'}), 404
            
        close_prices = stock_data['Close'].values.reshape(-1, 1)
        
        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        
        if len(scaled_data) < LOOK_BACK:
            return jsonify({'error': f'Not enough data points. Need at least {LOOK_BACK} days'}), 400
            
        X, _ = create_dataset(scaled_data, LOOK_BACK)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Predict historical period
        predictions = models[ticker].predict(X, verbose=0)
        predictions = scaler.inverse_transform(predictions)
        
        # Predict future prices if requested
        future_predictions = []
        future_dates = []
        if future_days > 0:
            # Get the last sequence from historical data
            last_sequence = scaled_data[-LOOK_BACK:]
            future_predictions = predict_future_prices(models[ticker], last_sequence, scaler, future_days)
            
            # Generate future dates
            last_date = stock_data.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=future_days
            ).strftime('%Y-%m-%d').tolist()
        
        # Create dates for historical predictions
        prediction_dates = pd.date_range(
            start=start_dt + timedelta(days=LOOK_BACK),
            periods=len(predictions)
        ).strftime('%Y-%m-%d').tolist()
        
        response = {
            'ticker': ticker,
            'historical': {
                'dates': prediction_dates,
                'actual': close_prices[LOOK_BACK:].flatten().tolist(),
                'predicted': predictions.flatten().tolist(),
            },
            'look_back_days': LOOK_BACK
        }
        
        if future_days > 0:
            response['future'] = {
                'dates': future_dates,
                'predicted': future_predictions.tolist()
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'running',
        'available_models': {ticker: ticker in models for ticker in SUPPORTED_TICKERS},
        'supported_tickers': SUPPORTED_TICKERS
    })

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise OSError(f"No available ports found in range {start_port}-{start_port + max_attempts - 1}")

if __name__ == '__main__':
    try:
        # Create models directory if needed
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Configure Flask logging
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        
        # Find available port
        port = find_available_port(DEFAULT_PORT, MAX_PORT_ATTEMPTS)
        if port != DEFAULT_PORT:
            logger.warning(f"Port {DEFAULT_PORT} in use, using port {port} instead")
        
        # Startup message
        logger.info("\n" + "="*50)
        logger.info("Stock Prediction API Server")
        logger.info("Available models:")
        for ticker in SUPPORTED_TICKERS:
            status = "✓" if ticker in models else "✗"
            logger.info(f"  {status} {ticker}")
        logger.info(f"\nServer running on http://127.0.0.1:{port}")
        logger.info("="*50 + "\n")
        
        # Run server
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise