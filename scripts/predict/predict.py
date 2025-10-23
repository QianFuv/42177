"""
Main pipeline for Cobb angle prediction using curve fitting and XGBoost.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from .feature_extractor import CurveFeatureExtractor
from .model import CobbAnglePredictor


def parse_bbox_string(bbox_str: str) -> np.ndarray:
    """
    Parse bounding box string from CSV format.

    Args:
        bbox_str: String containing bounding boxes in format "x,y,w,h;x,y,w,h;..."

    Returns:
        Array of bounding boxes with shape (N, 4)
    """
    if pd.isna(bbox_str) or bbox_str == '':
        return np.array([])

    bbox_list = []
    for bbox in bbox_str.split(';'):
        coords = [float(x) for x in bbox.split(',')]
        if len(coords) == 4:
            x, y, w, h = coords
            center_x = x + w / 2
            center_y = y + h / 2
            bbox_list.append([center_x, center_y, w, h])

    return np.array(bbox_list)


def load_data_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Loaded DataFrame
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    return df


def extract_features_from_dataframe(
    df: pd.DataFrame,
    feature_extractor: CurveFeatureExtractor
) -> tuple[np.ndarray, list[int]]:
    """
    Extract curve features from all samples in dataframe.

    Args:
        df: DataFrame containing bboxes column
        feature_extractor: Instance of CurveFeatureExtractor

    Returns:
        Tuple of (feature matrix, valid indices)
    """
    print("Extracting curve features from bounding boxes...")

    features_list = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        bboxes = parse_bbox_string(row.get('bboxes', ''))

        if len(bboxes) == 0:
            continue

        feature_vector = feature_extractor.extract_features(bboxes)

        if feature_vector is not None:
            features_list.append(feature_vector)
            valid_indices.append(idx)

    if len(features_list) == 0:
        raise ValueError("No valid features extracted from dataset")

    features_array = np.vstack(features_list)
    print(f"Extracted features from {len(features_list)} samples")

    return features_array, valid_indices


def prepare_targets(df: pd.DataFrame, indices: list[int]) -> np.ndarray:
    """
    Prepare target Cobb angles from dataframe.

    Args:
        df: DataFrame containing Cobb angle columns
        indices: List of valid sample indices

    Returns:
        Array of target angles with shape (N, 3)
    """
    angle_columns = ['cobb_angle_1', 'cobb_angle_2', 'cobb_angle_3']

    targets = df.loc[indices, angle_columns].values
    return targets


def train_model(
    csv_path: Path,
    output_dir: Path,
    poly_degree: int = 7,
    spline_smoothing: float = 1.0,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    validation_split: float = 0.2
) -> None:
    """
    Train Cobb angle prediction model.

    Args:
        csv_path: Path to training data CSV
        output_dir: Directory to save trained model
        poly_degree: Polynomial degree for curve fitting
        spline_smoothing: Smoothing factor for spline interpolation
        n_estimators: Number of XGBoost estimators
        max_depth: Maximum tree depth
        learning_rate: Learning rate for XGBoost
        validation_split: Fraction of data for validation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data_from_csv(csv_path)

    feature_extractor = CurveFeatureExtractor(
        poly_degree=poly_degree,
        spline_smoothing=spline_smoothing
    )

    X, valid_indices = extract_features_from_dataframe(df, feature_extractor)
    y = prepare_targets(df, valid_indices)

    feature_names = feature_extractor.get_feature_names()

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")

    predictor = CobbAnglePredictor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )

    print("\nTraining XGBoost model...")
    metrics = predictor.train(
        X, y,
        feature_names=feature_names,
        validation_split=validation_split
    )

    predictor.print_metrics(metrics)

    model_path = output_dir / 'cobb_angle_predictor.pkl'
    predictor.save(model_path)
    print(f"Model saved to {model_path}")

    importance_df = predictor.get_feature_importance()
    if importance_df is not None:
        importance_path = output_dir / 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        print(f"Feature importance saved to {importance_path}")

        print("\nTop 10 most important features:")
        top_features = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False).head(10)
        for feature, importance in top_features.items():
            print(f"  {feature:<30}: {importance:.4f}")

    metrics_path = output_dir / 'training_metrics.csv'
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Training metrics saved to {metrics_path}")


def predict_cobb_angles(
    model_path: Path,
    csv_path: Path,
    output_path: Path,
    poly_degree: int = 7,
    spline_smoothing: float = 1.0
) -> None:
    """
    Predict Cobb angles using trained model.

    Args:
        model_path: Path to trained model file
        csv_path: Path to input data CSV
        output_path: Path to save predictions
        poly_degree: Polynomial degree for curve fitting
        spline_smoothing: Smoothing factor for spline interpolation
    """
    df = load_data_from_csv(csv_path)

    print(f"Loading model from {model_path}...")
    predictor = CobbAnglePredictor()
    predictor.load(model_path)

    feature_extractor = CurveFeatureExtractor(
        poly_degree=poly_degree,
        spline_smoothing=spline_smoothing
    )

    X, valid_indices = extract_features_from_dataframe(df, feature_extractor)

    print("\nPredicting Cobb angles...")
    predictions = predictor.predict(X)

    results_df = df.loc[valid_indices].copy()
    results_df['predicted_angle_1'] = predictions[:, 0]
    results_df['predicted_angle_2'] = predictions[:, 1]
    results_df['predicted_angle_3'] = predictions[:, 2]

    if all(col in df.columns for col in ['cobb_angle_1', 'cobb_angle_2', 'cobb_angle_3']):
        true_angles = results_df[['cobb_angle_1', 'cobb_angle_2', 'cobb_angle_3']].values

        metrics = predictor.evaluate(X, true_angles)
        predictor.print_metrics(metrics)

        results_df['error_angle_1'] = results_df['cobb_angle_1'] - results_df['predicted_angle_1']
        results_df['error_angle_2'] = results_df['cobb_angle_2'] - results_df['predicted_angle_2']
        results_df['error_angle_3'] = results_df['cobb_angle_3'] - results_df['predicted_angle_3']

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")


def main():
    """
    Main entry point for the prediction pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Cobb angle prediction using curve fitting and XGBoost'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    train_parser = subparsers.add_parser('train', help='Train prediction model')
    train_parser.add_argument(
        '--csv',
        type=str,
        default='data_index/data.csv',
        help='Path to training data CSV'
    )
    train_parser.add_argument(
        '--output',
        type=str,
        default='models/cobb_predictor',
        help='Output directory for trained model'
    )
    train_parser.add_argument(
        '--poly-degree',
        type=int,
        default=7,
        help='Polynomial degree for curve fitting'
    )
    train_parser.add_argument(
        '--spline-smoothing',
        type=float,
        default=1.0,
        help='Smoothing factor for spline interpolation'
    )
    train_parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of XGBoost estimators'
    )
    train_parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Maximum tree depth'
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate'
    )
    train_parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )

    predict_parser = subparsers.add_parser('predict', help='Predict Cobb angles')
    predict_parser.add_argument(
        '--model',
        type=str,
        default='models/cobb_predictor/cobb_angle_predictor.pkl',
        help='Path to trained model'
    )
    predict_parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to input data CSV'
    )
    predict_parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Path to save predictions'
    )
    predict_parser.add_argument(
        '--poly-degree',
        type=int,
        default=7,
        help='Polynomial degree for curve fitting'
    )
    predict_parser.add_argument(
        '--spline-smoothing',
        type=float,
        default=1.0,
        help='Smoothing factor for spline interpolation'
    )

    args = parser.parse_args()

    if args.command == 'train':
        train_model(
            csv_path=Path(args.csv),
            output_dir=Path(args.output),
            poly_degree=args.poly_degree,
            spline_smoothing=args.spline_smoothing,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            validation_split=args.val_split
        )
    elif args.command == 'predict':
        predict_cobb_angles(
            model_path=Path(args.model),
            csv_path=Path(args.csv),
            output_path=Path(args.output),
            poly_degree=args.poly_degree,
            spline_smoothing=args.spline_smoothing
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
