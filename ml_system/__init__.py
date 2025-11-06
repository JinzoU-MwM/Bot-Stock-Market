"""
Machine Learning Enhancement System for Stock Signal Analysis

This module provides ML-powered signal enhancement including:
- Random Forest signal classification
- Technical indicator feature engineering
- Confidence scoring and risk management
- Production-ready integration with main trading system

Version: 1.0.0
Author: ML Enhancement Team
"""

__version__ = "1.0.0"
__author__ = "ML Enhancement Team"

from .core.ml_predictor import MLPredictor
from .core import *

__all__ = [
    'MLPredictor',
]