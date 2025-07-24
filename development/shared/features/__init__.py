"""
Feature Engineering Module for FinRL Bitcoin Trading

This module provides comprehensive feature extraction capabilities including:
- Technical indicators via pandas-ta
- Limit Order Book (LOB) specific features
- Feature selection and dimensionality reduction
- Integration with TradeSimulator
"""

from .technical_indicators import TechnicalIndicators
from .lob_features import LOBFeatures
from .feature_selector import FeatureSelector
from .feature_processor import FeatureProcessor

__all__ = [
    'TechnicalIndicators',
    'LOBFeatures', 
    'FeatureSelector',
    'FeatureProcessor'
]