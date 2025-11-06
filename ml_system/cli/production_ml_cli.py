#!/usr/bin/env python3
"""
ML Enhanced Stock Signal CLI - Production Version

Simple, robust CLI that works without unicode/emoji issues.
Focus on real implementation without Rich library dependencies.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import yfinance as yf
import joblib

class ProductionMLAnalyzer:
    """Production-ready ML analyzer for stock signals."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = []
        self.reverse_mapping = {}
        self.load_model()

    def load_model(self):
        """Load the trained Random Forest model."""
        try:
            if os.path.exists('models/working_rf_model.pkl'):
                self.model = joblib.load('models/working_rf_model.pkl')
                self.scaler = joblib.load('models/working_rf_scaler.pkl')
                model_info = joblib.load('models/working_rf_info.pkl')
                self.features = model_info['features']
                self.reverse_mapping = model_info['reverse_mapping']
                print(f"[SUCCESS] ML Model loaded with {len(self.features)} features")
                return True
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
        return False

    def is_loaded(self):
        return self.model is not None

    def create_features(self, df):
        """Create features for prediction."""
        if df.empty or len(df) < 30:
            return pd.DataFrame()

        features = df.copy()
        close = features['Close']

        try:
            # Basic features
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
            print(f"[ERROR] Error creating features: {e}")
            return pd.DataFrame()

    def predict_signal(self, symbol):
        """Predict signal for a stock symbol."""
        try:
            print(f"[INFO] Analyzing {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")

            if data.empty:
                return {
                    'symbol': symbol,
                    'signal': 'WAIT',
                    'confidence': 0.0,
                    'error': 'No data available'
                }

            # Create features
            features = self.create_features(data)

            if features.empty:
                return {
                    'symbol': symbol,
                    'signal': 'WAIT',
                    'confidence': 0.0,
                    'error': 'Could not create features'
                }

            # Get latest features
            latest_features = features.iloc[[-1]][self.features].fillna(0)
            current_price = data['Close'].iloc[-1]

            # Scale and predict
            scaled_features = self.scaler.transform(latest_features)
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]

            # Get signal and confidence
            signal = self.reverse_mapping.get(prediction, 'WAIT')
            confidence = probabilities[np.where(self.model.classes_ == prediction)[0][0]]

            return {
                'symbol': symbol,
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
            return {
                'symbol': symbol,
                'signal': 'WAIT',
                'confidence': 0.0,
                'error': str(e),
                'success': False
            }

def print_header():
    print("=" * 60)
    print("         ML ENHANCED STOCK SIGNAL ANALYZER")
    print("=" * 60)

def print_results(results):
    """Print analysis results in clean format."""
    print("\n" + "=" * 80)
    print(f"{'SYMBOL':<12} {'SIGNAL':<8} {'CONFIDENCE':<12} {'PRICE':<15} {'STATUS'}")
    print("-" * 80)

    for result in results:
        if result['success']:
            signal_color = result['signal']
            conf_color = "HIGH" if result['confidence'] > 0.6 else "MED" if result['confidence'] > 0.4 else "LOW"

            print(f"{result['symbol']:<12} {result['signal']:<8} {result['confidence']:<12.3f} {result['current_price']:<15.2f} {conf_color}")
        else:
            print(f"{result['symbol']:<12} {'ERROR':<8} {'0.000':<12} {'-':<15} FAILED")

def main():
    """Main production CLI function."""
    print_header()

    # Initialize ML analyzer
    ml_analyzer = ProductionMLAnalyzer()

    # Show status
    if ml_analyzer.is_loaded():
        print(f"[STATUS] ML System: ACTIVE with {len(ml_analyzer.features)} features")
    else:
        print("[STATUS] ML System: INACTIVE - Using fallback analysis")

    print("\nCommands:")
    print("  - Enter stock symbols (e.g., BBCA BBRI TLKM)")
    print("  - 'status' for ML system status")
    print("  - 'exit' to quit")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nEnter symbols or command: ").strip()

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in {"exit", "quit", "q"}:
                print("[INFO] Exiting program...")
                break

            if cmd == "status":
                if ml_analyzer.is_loaded():
                    print(f"[ACTIVE] ML Model Status: Operational")
                    print(f"[INFO] Features: {len(ml_analyzer.features)}")
                    print(f"[INFO] Signal Classes: {ml_analyzer.reverse_mapping}")
                else:
                    print("[INACTIVE] ML Model Status: Not Available")
                    print("[INFO] Model files not found in models/ directory")
                continue

            # Treat everything else as stock symbols
            symbols = user_input.upper().split()
            results = []

            print(f"\n[PROCESSING] Analyzing {len(symbols)} symbols...")

            for symbol in symbols:
                if not symbol.endswith('.JK'):
                    symbol += '.JK'
                result = ml_analyzer.predict_signal(symbol)
                results.append(result)

            # Display results
            print_results(results)

            # Summary
            successful = sum(1 for r in results if r['success'])
            print(f"\n[SUMMARY] {successful}/{len(results)} analyses completed successfully")

        except KeyboardInterrupt:
            print("\n[INFO] Exiting program...")
            break
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()