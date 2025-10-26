"""
XGBoost regression model for Cobb angle prediction.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, cast
from numpy.typing import NDArray
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor


class CobbAnglePredictor:
    """
    XGBoost-based multi-output regression model for predicting three Cobb angles.

    This class handles training, evaluation, and inference of Cobb angle predictions
    from spine curve features.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **xgb_params
    ):
        """
        Initialize the Cobb angle predictor.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate for boosting
            random_state: Random seed for reproducibility
            **xgb_params: Additional XGBoost parameters
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.xgb_params = xgb_params

        self.model = None
        self.feature_names = None
        self.is_fitted = False

    def _create_base_model(self) -> XGBRegressor:
        """
        Create a base XGBoost regressor with configured parameters.

        Returns:
            Configured XGBRegressor instance
        """
        return XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            **self.xgb_params
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the multi-output regression model.

        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Target matrix with shape (n_samples, 3) for three Cobb angles
            feature_names: List of feature names
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary containing training metrics
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=self.random_state
        )

        self.model = MultiOutputRegressor(self._create_base_model())
        self.model.fit(X_train, y_train)

        self.feature_names = feature_names
        self.is_fitted = True

        train_metrics = self.evaluate(X_train, y_train, prefix='train')
        val_metrics = self.evaluate(X_val, y_val, prefix='val')

        metrics = {**train_metrics, **val_metrics}
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict Cobb angles from features.

        Args:
            X: Feature matrix with shape (n_samples, n_features)

        Returns:
            Predicted Cobb angles with shape (n_samples, 3)

        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        predictions = self.model.predict(X)
        predictions_array = np.asarray(predictions)

        if predictions_array.ndim == 1:
            predictions_array = predictions_array.reshape(-1, 1)

        return predictions_array

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Evaluate model performance on given data.

        Args:
            X: Feature matrix
            y_true: True target values
            prefix: Prefix for metric names

        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X)

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        metrics = {}

        for i in range(y_true.shape[1]):
            angle_name = f'angle_{i + 1}'
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true[:, i], y_pred[:, i])

            mape = self._calculate_mape(y_true[:, i], y_pred[:, i])

            caa = self._calculate_caa(y_true[:, i], y_pred[:, i], threshold=5.0)

            if prefix:
                angle_name = f'{prefix}_{angle_name}'

            metrics[f'{angle_name}_mae'] = mae
            metrics[f'{angle_name}_rmse'] = rmse
            metrics[f'{angle_name}_r2'] = r2
            metrics[f'{angle_name}_mape'] = mape
            metrics[f'{angle_name}_caa'] = caa

        overall_mae = mean_absolute_error(y_true, y_pred)
        overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        overall_mape = self._calculate_mape(y_true.flatten(), y_pred.flatten())
        overall_caa = self._calculate_caa(y_true.flatten(), y_pred.flatten(), threshold=5.0)

        if prefix:
            metrics[f'{prefix}_overall_mae'] = overall_mae
            metrics[f'{prefix}_overall_rmse'] = overall_rmse
            metrics[f'{prefix}_overall_mape'] = overall_mape
            metrics[f'{prefix}_overall_caa'] = overall_caa
        else:
            metrics['overall_mae'] = overall_mae
            metrics['overall_rmse'] = overall_rmse
            metrics['overall_mape'] = overall_mape
            metrics['overall_caa'] = overall_caa

        return metrics

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            MAPE value as percentage
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        non_zero_mask = y_true != 0
        if not np.any(non_zero_mask):
            return 0.0

        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        return float(mape)

    def _calculate_caa(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 5.0) -> float:
        """
        Calculate Clinically Acceptable Accuracy (CAA).

        Args:
            y_true: True values
            y_pred: Predicted values
            threshold: Acceptable error threshold in degrees

        Returns:
            CAA value as percentage
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        errors = np.abs(y_true - y_pred)
        acceptable = np.sum(errors <= threshold)
        total = len(errors)

        caa = (acceptable / total) * 100 if total > 0 else 0.0
        return float(caa)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores from the trained model.

        Returns:
            DataFrame with feature names and importance scores, or None if not fitted
        """
        if not self.is_fitted or self.feature_names is None or self.model is None:
            return None

        if not hasattr(self.model, 'estimators_'):
            return None

        importance_list = []
        for i, estimator in enumerate(self.model.estimators_):
            if hasattr(estimator, 'feature_importances_'):
                importance = getattr(estimator, 'feature_importances_')
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance,
                    'target': f'cobb_angle_{i + 1}'
                })
                importance_list.append(importance_df)

        if not importance_list:
            return None

        combined_importance = pd.concat(importance_list, ignore_index=True)
        return combined_importance

    def save(self, path: Path) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model file

        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'xgb_params': self.xgb_params,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, path: Path) -> None:
        """
        Load a trained model from disk.

        Args:
            path: Path to the model file

        Raises:
            FileNotFoundError: If model file does not exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.n_estimators = model_data['n_estimators']
        self.max_depth = model_data['max_depth']
        self.learning_rate = model_data['learning_rate']
        self.random_state = model_data['random_state']
        self.xgb_params = model_data['xgb_params']
        self.is_fitted = True

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Print evaluation metrics in a formatted table.

        Args:
            metrics: Dictionary containing evaluation metrics
        """
        print("\n" + "=" * 60)
        print("Model Evaluation Metrics")
        print("=" * 60)

        for key, value in sorted(metrics.items()):
            print(f"{key:<30}: {value:>10.4f}")

        print("=" * 60 + "\n")
