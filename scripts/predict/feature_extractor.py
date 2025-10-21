"""
Curve feature extraction from vertebrae bounding boxes.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import trapezoid
from typing import Dict, Tuple, Optional


class CurveFeatureExtractor:
    """
    Extract spine curve features from vertebrae bounding boxes for Cobb angle prediction.

    This class implements curve fitting using polynomial regression and spline interpolation
    to extract geometric features from detected vertebrae positions.
    """

    def __init__(
        self,
        poly_degree: int = 7,
        spline_smoothing: float = 1.0,
        num_sample_points: int = 100
    ):
        """
        Initialize the curve feature extractor.

        Args:
            poly_degree: Degree of polynomial for curve fitting (default: 7)
            spline_smoothing: Smoothing factor for spline interpolation (default: 1.0)
            num_sample_points: Number of points for curve sampling (default: 100)
        """
        self.poly_degree = poly_degree
        self.spline_smoothing = spline_smoothing
        self.num_sample_points = num_sample_points

    def extract_vertebra_centers(self, bboxes: np.ndarray) -> np.ndarray:
        """
        Extract center points from bounding boxes.

        Args:
            bboxes: Array of bounding boxes with shape (N, 4) containing [x_center, y_center, width, height]

        Returns:
            Array of center points with shape (N, 2) containing [x_center, y_center]
        """
        if bboxes.ndim != 2 or bboxes.shape[1] < 2:
            raise ValueError("Bboxes must be 2D array with at least 2 columns")

        return bboxes[:, :2]

    def sort_centers_by_vertical_position(self, centers: np.ndarray) -> np.ndarray:
        """
        Sort center points by vertical position (top to bottom).

        Args:
            centers: Array of center points with shape (N, 2)

        Returns:
            Sorted array of center points
        """
        sorted_indices = np.argsort(centers[:, 1])
        return centers[sorted_indices]

    def fit_polynomial_curve(
        self,
        centers: np.ndarray
    ) -> Tuple[np.ndarray, np.poly1d]:
        """
        Fit a polynomial curve to vertebra center points.

        Args:
            centers: Array of center points with shape (N, 2)

        Returns:
            Tuple of (polynomial coefficients, polynomial function)
        """
        x = centers[:, 1]
        y = centers[:, 0]

        coeffs = np.polyfit(x, y, deg=self.poly_degree)
        poly_func = np.poly1d(coeffs)

        return coeffs, poly_func

    def fit_spline_curve(self, centers: np.ndarray) -> UnivariateSpline:
        """
        Fit a smoothing spline to vertebra center points.

        Args:
            centers: Array of center points with shape (N, 2)

        Returns:
            UnivariateSpline object
        """
        x = centers[:, 1]
        y = centers[:, 0]

        spline = UnivariateSpline(x, y, s=self.spline_smoothing)
        return spline

    def compute_curvature(
        self,
        x_points: np.ndarray,
        y_points: np.ndarray
    ) -> np.ndarray:
        """
        Compute curvature along the fitted curve.

        Args:
            x_points: X coordinates along the curve
            y_points: Y coordinates along the curve

        Returns:
            Array of curvature values
        """
        dy = np.gradient(y_points, x_points)
        d2y = np.gradient(dy, x_points)

        curvature = np.abs(d2y) / np.power(1 + dy**2, 1.5)
        return curvature

    def compute_curve_length(
        self,
        x_points: np.ndarray,
        y_points: np.ndarray
    ) -> float:
        """
        Compute the arc length of the fitted curve.

        Args:
            x_points: X coordinates along the curve
            y_points: Y coordinates along the curve

        Returns:
            Total curve length
        """
        dy = np.gradient(y_points, x_points)
        integrand = np.sqrt(1 + dy**2)
        curve_length = trapezoid(integrand, x_points)

        return float(curve_length)

    def count_inflection_points(self, d2y: np.ndarray) -> int:
        """
        Count the number of inflection points in the curve.

        Args:
            d2y: Second derivative of the curve

        Returns:
            Number of inflection points
        """
        sign_changes = np.diff(np.sign(d2y))
        inflection_count = len(np.where(sign_changes != 0)[0])

        return inflection_count

    def extract_geometric_features(
        self,
        poly_func: np.poly1d,
        x_min: float,
        x_max: float
    ) -> Dict[str, float]:
        """
        Extract geometric features from the fitted polynomial curve.

        Args:
            poly_func: Fitted polynomial function
            x_min: Minimum x coordinate
            x_max: Maximum x coordinate

        Returns:
            Dictionary containing geometric features
        """
        x_dense = np.linspace(x_min, x_max, self.num_sample_points)
        y_fitted = poly_func(x_dense)

        dy = np.gradient(y_fitted, x_dense)
        d2y = np.gradient(dy, x_dense)

        curvature = self.compute_curvature(x_dense, y_fitted)
        curve_length = self.compute_curve_length(x_dense, y_fitted)
        inflection_count = self.count_inflection_points(d2y)

        features = {
            'max_curvature': float(np.max(curvature)),
            'mean_curvature': float(np.mean(curvature)),
            'std_curvature': float(np.std(curvature)),
            'curve_length': curve_length,
            'max_slope': float(np.max(np.abs(dy))),
            'mean_slope': float(np.mean(np.abs(dy))),
            'inflection_points': float(inflection_count),
            'curve_range_x': float(x_max - x_min),
            'curve_range_y': float(np.max(y_fitted) - np.min(y_fitted)),
        }

        return features

    def extract_features(self, bboxes: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract complete feature vector from vertebrae bounding boxes.

        Args:
            bboxes: Array of bounding boxes with shape (N, 4)

        Returns:
            Feature vector as 1D numpy array, or None if extraction fails
        """
        if len(bboxes) < self.poly_degree + 1:
            return None

        centers = self.extract_vertebra_centers(bboxes)
        centers = self.sort_centers_by_vertical_position(centers)

        poly_coeffs, poly_func = self.fit_polynomial_curve(centers)

        x_min = centers[:, 1].min()
        x_max = centers[:, 1].max()

        geometric_features = self.extract_geometric_features(poly_func, x_min, x_max)

        feature_vector = np.concatenate([
            poly_coeffs,
            [
                geometric_features['max_curvature'],
                geometric_features['mean_curvature'],
                geometric_features['std_curvature'],
                geometric_features['curve_length'],
                geometric_features['max_slope'],
                geometric_features['mean_slope'],
                geometric_features['inflection_points'],
                geometric_features['curve_range_x'],
                geometric_features['curve_range_y'],
            ]
        ])

        return feature_vector

    def get_feature_names(self) -> list[str]:
        """
        Get names of all extracted features.

        Returns:
            List of feature names
        """
        poly_names = [f'poly_coeff_{i}' for i in range(self.poly_degree + 1)]
        geometric_names = [
            'max_curvature',
            'mean_curvature',
            'std_curvature',
            'curve_length',
            'max_slope',
            'mean_slope',
            'inflection_points',
            'curve_range_x',
            'curve_range_y',
        ]

        return poly_names + geometric_names
