"""
Package src.models:
Chứa các kiến trúc mô hình dự báo giá vàng:
- baselines.py: Naive Persistence & Linear/Ridge Regression
- tree_models.py: XGBoost & LightGBM Regressor
- lstm_model.py: PyTorch Multi-variable LSTM
"""

from .baselines import (
    BaseForecaster,
    NaivePersistenceModel,
    LinearRegressionBaseline,
)

__all__ = [
    "BaseForecaster",
    "NaivePersistenceModel",
    "LinearRegressionBaseline",
]
