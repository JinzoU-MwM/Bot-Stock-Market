#!/usr/bin/env python3
"""
Test ML Integration with New Organized Structure
"""

import sys
import os
import pandas as pd
import yfinance as yf
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analyzers.market_analyzer import MarketAnalyzer
from ml_system.core import MLPredictor

def test_organized_integration():
    print("=" * 60)
    print("    TESTING ML INTEGRATION WITH ORGANIZED STRUCTURE")
    print("=" * 60)

    # Test ML System Direct
    print("\n1. Testing ML System Direct:")
    ml_predictor = MLPredictor()
    print(f"   ML Status: {'ENABLED' if ml_predictor.is_enabled() else 'DISABLED'}")

    # Test Integration with MarketAnalyzer
    print("\n2. Testing MarketAnalyzer Integration:")
    analyzer = MarketAnalyzer()

    # Test with Indonesian stock
    symbol = "BBCA.JK"
    print(f"\n3. Testing {symbol}...")

    try:
        # Get data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="3mo")

        if data.empty:
            print("   No data available")
            return

        # Rename columns to lowercase for analyzer compatibility
        data.columns = [col.lower() for col in data.columns]

        print(f"   Data shape: {data.shape}")

        # Analyze with ML integration
        analysis = analyzer.analyze_market(data, symbol)

        # Check ML analysis
        ml_analysis = analysis.get('ml', {})
        overall = analysis.get('overall', {})

        print(f"   ML Signal: {ml_analysis.get('signal', 'N/A')} (confidence: {ml_analysis.get('confidence', 0):.3f})")
        print(f"   Overall Signal: {overall.get('signal', 'N/A')} (strength: {overall.get('strength', 0):.3f})")

        # Check ML status
        if ml_analysis.get('available', False):
            print(f"   [SUCCESS] ML Integration: WORKING")
        else:
            print(f"   [FAILED] ML Integration: {ml_analysis.get('reason', 'Unknown')}")

    except Exception as e:
        print(f"   Error: {e}")

    print(f"\n{'='*60}")
    print("ORGANIZED ML STRUCTURE TEST COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    test_organized_integration()