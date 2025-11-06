# ML System Architecture

## 🏗️ System Architecture Overview

```
Bot-Stock-Market/
├── scripts/
│   └── stock_signal_cli.py        # Main CLI Application
├── analyzers/
│   └── market_analyzer.py         # Enhanced with ML integration
├── ml_system/                     # Organized ML Enhancement
│   ├── core/
│   │   ├── __init__.py
│   │   └── ml_predictor.py        # Main ML predictor class
│   ├── models/                    # Trained model files
│   │   ├── working_rf_model.pkl   # Random Forest model (711KB)
│   │   ├── working_rf_scaler.pkl  # Feature scaler
│   │   └── working_rf_info.pkl    # Model metadata
│   ├── cli/
│   │   └── production_ml_cli.py   # Standalone ML CLI
│   ├── tests/
│   │   └── test_integration.py    # Integration tests
│   ├── docs/                      # Documentation
│   ├── data/                      # Training/test data
│   └── README.md
```

## 🔄 Data Flow

```
User Input (stock symbols)
         ↓
stock_signal_cli.py
         ↓
MarketAnalyzer (Enhanced)
         ↓
┌─────────────────────────────────┐
│     Traditional Analysis        │
│  ───────────────────────────── │
│  • Technical Indicators         │
│  • Pattern Recognition          │
│  • Breakout Detection           │
│  • Support/Resistance           │
│  • Scalping Signals             │
│  • News Analysis                │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│        ML Enhancement           │
│  ───────────────────────────── │
│  • MLPredictor                  │
│  • 14 Technical Features        │
│  • Random Forest (64.5% acc)   │
│  • Confidence Scoring           │
└─────────────────────────────────┘
         ↓
Signal Combination (20% ML weight)
         ↓
Enhanced Trading Signal
         ↓
Output to User
```

## 🎯 ML Model Details

### Random Forest Classifier
- **Accuracy**: 64.5% (test), 98.9% (training)
- **Features**: 14 technical indicators
- **Classes**: BUY, SELL, WAIT signals
- **Confidence**: Probability-based scoring

### Feature Engineering
1. **Returns**: 5-day, 10-day, 20-day returns
2. **Moving Averages**: SMA and price ratios
3. **Volatility**: Rolling standard deviations
4. **RSI**: 14-period RSI calculation
5. **Momentum**: 5-day momentum percentage

### Signal Integration
- **ML Weight**: 20% in overall signal calculation
- **Confidence Weighting**: ML signal × confidence × 0.2
- **Fallback**: Traditional analysis when ML unavailable

## 🔧 Configuration

### Model Loading Priority
1. `ml_system/models/` (primary)
2. `.worktrees/ml-enhancement/models/` (fallback)

### Error Handling
- Graceful degradation when model unavailable
- Column name compatibility (Close/close)
- Data validation and cleaning
- Feature missing handling

## 📊 Performance Metrics

| Symbol | ML Signal | Confidence | Overall Signal | Strength |
|--------|-----------|-------------|----------------|----------|
| BBCA.JK| WAIT     | 79.3%       | BUY           | 0.250    |
| BBRI.JK| SELL     | 50.9%       | WAIT          | 0.002    |
| TLKM.JK| SELL     | 46.9%       | BUY           | 0.156    |

## 🚀 Usage Examples

### Basic Integration
```python
from analyzers.market_analyzer import MarketAnalyzer

analyzer = MarketAnalyzer()
analysis = analyzer.analyze_market(data, "BBCA.JK")
ml_signal = analysis['ml']['signal']
```

### Direct ML Usage
```python
from ml_system.core import MLPredictor

ml = MLPredictor()
prediction = ml.predict_signal(data, "BBCA.JK")
```

### CLI Usage
```bash
python ml_system/cli/production_ml_cli.py
```

---

*Architecture Version 1.0*