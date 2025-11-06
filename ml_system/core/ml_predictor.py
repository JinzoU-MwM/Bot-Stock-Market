"""
ML Predictor Integration
Bridge between ML system and main program
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from typing import Dict, Optional

class MLPredictor:
    """ML predictor for stock signals - Integrates with main system"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = []
        self.reverse_mapping = {}
        self.enabled = True
        self.load_model()

    def load_model(self):
        """Load ML model from ml_system models directory"""
        try:
            # Try ml_system/models first, then fallback to worktree
            model_locations = [
                os.path.join('ml_system', 'models'),
                os.path.join('.worktrees', 'ml-enhancement', 'models')
            ]

            for base_path in model_locations:
                model_file = os.path.join(base_path, 'working_rf_model.pkl')
                if os.path.exists(model_file):
                    self.model = joblib.load(os.path.join(base_path, 'working_rf_model.pkl'))
                    self.scaler = joblib.load(os.path.join(base_path, 'working_rf_scaler.pkl'))
                    model_info = joblib.load(os.path.join(base_path, 'working_rf_info.pkl'))
                    self.features = model_info['features']
                    self.reverse_mapping = model_info['reverse_mapping']
                    print(f"[ML] Model loaded from {base_path} with {len(self.features)} features")
                    return True

            print("[ML] Model not found in any location - ML features disabled")
            self.enabled = False
            return False
        except Exception as e:
            print(f"[ML] Error loading model: {e}")
            self.enabled = False
            return False

    def create_features(self, df):
        """Create ML features from market data"""
        if df.empty or len(df) < 30:
            return pd.DataFrame()

        features = df.copy()

        # Handle both 'Close' and 'close' column names
        if 'Close' in features.columns:
            close = features['Close']
        elif 'close' in features.columns:
            close = features['close']
        else:
            return pd.DataFrame()

        try:
            # Technical indicators matching ML training
            for period in [5, 10, 20]:
                features[f'return_{period}d'] = close.pct_change(period)
                features[f'sma_{period}'] = close.rolling(window=period).mean()
                features[f'price_to_sma_{period}'] = close / features[f'sma_{period}']
                features[f'volatility_{period}d'] = features[f'return_{period}d'].rolling(window=period).std()

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi_14'] = 100 - (100 / (1 + rs))

            # Momentum
            features['momentum_5'] = (close / close.shift(5) - 1) * 100

            # Clean data
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(0)

            return features
        except Exception as e:
            print(f"[ML] Error creating features: {e}")
            return pd.DataFrame()

    def predict_signal(self, df, symbol):
        """Predict signal using ML model"""
        if not self.enabled or self.model is None:
            return None

        try:
            # Create features
            features = self.create_features(df)

            if features.empty or not all(f in features.columns for f in self.features):
                return None

            # Get latest features
            latest_features = features.iloc[[-1]][self.features].fillna(0)

            # Handle both 'Close' and 'close' for current price
            if 'Close' in df.columns:
                current_price = df['Close'].iloc[-1]
            elif 'close' in df.columns:
                current_price = df['close'].iloc[-1]
            else:
                current_price = 0

            # Scale and predict
            scaled_features = self.scaler.transform(latest_features)
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]

            # Get signal and confidence
            signal = self.reverse_mapping.get(prediction, 'WAIT')
            confidence = probabilities[np.where(self.model.classes_ == prediction)[0][0]]

            return {
                'signal': signal,
                'confidence': confidence,
                'current_price': current_price,
                'success': True,
                'probabilities': {
                    self.reverse_mapping.get(cls, 'UNKNOWN'): float(prob)
                    for cls, prob in zip(self.model.classes_, probabilities)
                }
            }

        except Exception as e:
            print(f"[ML] Prediction error for {symbol}: {e}")
            return None

    def is_enabled(self):
        """Check if ML predictor is enabled"""
        return self.enabled and self.model is not None