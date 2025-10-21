"""
Cobb angle prediction module using curve fitting and XGBoost regression.
"""

from .feature_extractor import CurveFeatureExtractor
from .model import CobbAnglePredictor
from .predict import main

__all__ = ['CurveFeatureExtractor', 'CobbAnglePredictor', 'main']
