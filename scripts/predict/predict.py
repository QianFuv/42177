"""
Main pipeline for Cobb angle prediction using curve fitting and XGBoost.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)

from .feature_extractor import CurveFeatureExtractor
from .model import CobbAnglePredictor


def classify_severity(angles: np.ndarray) -> np.ndarray:
    """
    Classify spinal scoliosis severity based on maximum Cobb angle.

    Args:
        angles: Array of Cobb angles with shape (N, 3) or (3,)

    Returns:
        Array of severity classifications: 'normal', 'mild', 'moderate', or 'severe'
    """
    if angles.ndim == 1:
        max_angle = np.max(angles)
    else:
        max_angle = np.max(angles, axis=1)

    severity = np.empty(max_angle.shape, dtype=object)
    severity[max_angle < 10] = 'normal'
    severity[(max_angle >= 10) & (max_angle < 25)] = 'mild'
    severity[(max_angle >= 25) & (max_angle <= 45)] = 'moderate'
    severity[max_angle > 45] = 'severe'

    return severity


def calculate_weighted_clinical_error(cm: np.ndarray) -> float:
    """
    Calculate Weighted Clinical Error (WCE) based on clinical severity.

    Args:
        cm: Confusion matrix with shape (4, 4) for ['normal', 'mild', 'moderate', 'severe']

    Returns:
        Weighted clinical error score
    """
    weight_matrix = np.array([
        [0, 1, 2, 3],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [3, 2, 1, 0]
    ])

    weighted_errors = cm * weight_matrix
    total_weighted_error = np.sum(weighted_errors)
    total_samples = np.sum(cm)

    wce = total_weighted_error / total_samples if total_samples > 0 else 0.0
    return float(wce)


def evaluate_classification(y_true_classes: np.ndarray, y_pred_classes: np.ndarray) -> dict:
    """
    Evaluate classification performance.

    Args:
        y_true_classes: True severity classes
        y_pred_classes: Predicted severity classes

    Returns:
        Dictionary containing classification metrics
    """
    labels = ['normal', 'mild', 'moderate', 'severe']

    accuracy = accuracy_score(y_true_classes, y_pred_classes)

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_classes,
        y_pred_classes,
        labels=labels,
        average='weighted',
        zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_classes,
        y_pred_classes,
        labels=labels,
        average='macro',
        zero_division=0
    )

    kappa = cohen_kappa_score(y_true_classes, y_pred_classes, labels=labels)

    cm = confusion_matrix(
        y_true_classes,
        y_pred_classes,
        labels=labels
    )

    wce = calculate_weighted_clinical_error(cm)

    report = classification_report(
        y_true_classes,
        y_pred_classes,
        labels=labels,
        zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_score_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_score_macro': f1_macro,
        'cohen_kappa': kappa,
        'weighted_clinical_error': wce,
        'confusion_matrix': cm,
        'classification_report': report
    }


def print_classification_metrics(metrics: dict, dataset_name: str = "Dataset") -> None:
    """
    Print classification evaluation metrics.

    Args:
        metrics: Dictionary containing classification metrics
        dataset_name: Name of the dataset being evaluated
    """
    print("\n" + "=" * 60)
    print(f"{dataset_name} Classification Metrics")
    print("=" * 60)
    print(f"{'accuracy':<30}: {metrics['accuracy']:>10.4f}")
    print("\nWeighted Metrics (account for class imbalance):")
    print(f"{'precision_weighted':<30}: {metrics['precision_weighted']:>10.4f}")
    print(f"{'recall_weighted':<30}: {metrics['recall_weighted']:>10.4f}")
    print(f"{'f1_score_weighted':<30}: {metrics['f1_score_weighted']:>10.4f}")
    print("\nMacro-averaged Metrics (treat all classes equally):")
    print(f"{'precision_macro':<30}: {metrics['precision_macro']:>10.4f}")
    print(f"{'recall_macro':<30}: {metrics['recall_macro']:>10.4f}")
    print(f"{'f1_score_macro':<30}: {metrics['f1_score_macro']:>10.4f}")
    print("\nAdvanced Metrics:")
    print(f"{'cohen_kappa':<30}: {metrics['cohen_kappa']:>10.4f}")
    print(f"{'weighted_clinical_error':<30}: {metrics['weighted_clinical_error']:>10.4f}")
    print("=" * 60)

    print(f"\nConfusion Matrix ({dataset_name}):")
    print("=" * 60)
    print(f"{'True \\ Predicted':<15} {'normal':>10} {'mild':>10} {'moderate':>10} {'severe':>10}")
    print("-" * 60)
    cm = metrics['confusion_matrix']
    labels = ['normal', 'mild', 'moderate', 'severe']
    for i, label in enumerate(labels):
        print(f"{label:<15} {cm[i][0]:>10d} {cm[i][1]:>10d} {cm[i][2]:>10d} {cm[i][3]:>10d}")
    print("=" * 60)

    print(f"\nDetailed Classification Report ({dataset_name}):")
    print("=" * 60)
    print(metrics['classification_report'])
    print("=" * 60)


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


def load_config_from_json(config_path: Path) -> dict:
    """
    Load hyperparameters from JSON configuration file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Dictionary containing hyperparameters
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def train_model(
    csv_path: Path,
    output_dir: Path,
    poly_degree: int = 7,
    spline_smoothing: float = 1.0,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    validation_split: float = 0.2,
    min_child_weight: int = 1,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    gamma: float = 0.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0
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
        min_child_weight: Minimum sum of instance weight needed in a child
        subsample: Subsample ratio of training instances
        colsample_bytree: Subsample ratio of columns when constructing each tree
        gamma: Minimum loss reduction required to make a split
        reg_alpha: L1 regularization term on weights
        reg_lambda: L2 regularization term on weights
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
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda
    )

    print("\nTraining XGBoost model...")
    metrics = predictor.train(
        X, y,
        feature_names=feature_names,
        validation_split=validation_split
    )

    predictor.print_metrics(metrics)

    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    y_train_pred = predictor.predict(X_train)
    y_val_pred = predictor.predict(X_val)

    y_train_classes = classify_severity(y_train)
    y_train_pred_classes = classify_severity(y_train_pred)
    y_val_classes = classify_severity(y_val)
    y_val_pred_classes = classify_severity(y_val_pred)

    train_class_metrics = evaluate_classification(y_train_classes, y_train_pred_classes)
    val_class_metrics = evaluate_classification(y_val_classes, y_val_pred_classes)

    print_classification_metrics(train_class_metrics, "Training")
    print_classification_metrics(val_class_metrics, "Validation")

    importance_df = predictor.get_feature_importance()
    if importance_df is not None:
        print("\n" + "=" * 60)
        print("Top 10 Most Important Features")
        print("=" * 60)
        top_features = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False).head(10)
        for feature, importance in top_features.items():
            print(f"{feature:<30}: {importance:>10.4f}")
        print("=" * 60)

    model_path = output_dir / 'cobb_angle_predictor.pkl'
    predictor.save(model_path)

    metrics_path = output_dir / 'training_metrics.csv'
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    class_metrics_data = {
        'dataset': ['train', 'validation'],
        'accuracy': [train_class_metrics['accuracy'], val_class_metrics['accuracy']],
        'precision_weighted': [train_class_metrics['precision_weighted'], val_class_metrics['precision_weighted']],
        'recall_weighted': [train_class_metrics['recall_weighted'], val_class_metrics['recall_weighted']],
        'f1_score_weighted': [train_class_metrics['f1_score_weighted'], val_class_metrics['f1_score_weighted']],
        'precision_macro': [train_class_metrics['precision_macro'], val_class_metrics['precision_macro']],
        'recall_macro': [train_class_metrics['recall_macro'], val_class_metrics['recall_macro']],
        'f1_score_macro': [train_class_metrics['f1_score_macro'], val_class_metrics['f1_score_macro']],
        'cohen_kappa': [train_class_metrics['cohen_kappa'], val_class_metrics['cohen_kappa']],
        'weighted_clinical_error': [train_class_metrics['weighted_clinical_error'], val_class_metrics['weighted_clinical_error']]
    }
    class_metrics_path = output_dir / 'classification_metrics.csv'
    pd.DataFrame(class_metrics_data).to_csv(class_metrics_path, index=False)

    train_cm_path = output_dir / 'train_confusion_matrix.csv'
    pd.DataFrame(
        train_class_metrics['confusion_matrix'],
        index=['normal', 'mild', 'moderate', 'severe'],
        columns=['normal', 'mild', 'moderate', 'severe']
    ).to_csv(train_cm_path)

    val_cm_path = output_dir / 'val_confusion_matrix.csv'
    pd.DataFrame(
        val_class_metrics['confusion_matrix'],
        index=['normal', 'mild', 'moderate', 'severe'],
        columns=['normal', 'mild', 'moderate', 'severe']
    ).to_csv(val_cm_path)

    importance_path = None
    if importance_df is not None:
        importance_path = output_dir / 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)

    print("\n" + "=" * 60)
    print("Training Complete - Files Saved")
    print("=" * 60)
    print(f"{'Model':<30}: {model_path}")
    print(f"{'Regression metrics':<30}: {metrics_path}")
    print(f"{'Classification metrics':<30}: {class_metrics_path}")
    print(f"{'Training confusion matrix':<30}: {train_cm_path}")
    print(f"{'Validation confusion matrix':<30}: {val_cm_path}")
    if importance_path is not None:
        print(f"{'Feature importance':<30}: {importance_path}")
    print("=" * 60)


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

    predicted_classes = classify_severity(predictions)
    results_df['predicted_severity_class'] = predicted_classes

    if all(col in df.columns for col in ['cobb_angle_1', 'cobb_angle_2', 'cobb_angle_3']):
        true_angles = results_df[['cobb_angle_1', 'cobb_angle_2', 'cobb_angle_3']].values

        metrics = predictor.evaluate(X, true_angles)
        predictor.print_metrics(metrics)

        results_df['error_angle_1'] = results_df['cobb_angle_1'] - results_df['predicted_angle_1']
        results_df['error_angle_2'] = results_df['cobb_angle_2'] - results_df['predicted_angle_2']
        results_df['error_angle_3'] = results_df['cobb_angle_3'] - results_df['predicted_angle_3']

        if 'severity_class' in results_df.columns:
            true_classes = results_df['severity_class'].values.astype(str)
            class_metrics = evaluate_classification(true_classes, predicted_classes)
            print_classification_metrics(class_metrics, "Prediction")

            output_dir = Path(output_path).parent
            class_metrics_path = output_dir / 'prediction_classification_metrics.csv'
            class_metrics_data = {
                'accuracy': [class_metrics['accuracy']],
                'precision_weighted': [class_metrics['precision_weighted']],
                'recall_weighted': [class_metrics['recall_weighted']],
                'f1_score_weighted': [class_metrics['f1_score_weighted']],
                'precision_macro': [class_metrics['precision_macro']],
                'recall_macro': [class_metrics['recall_macro']],
                'f1_score_macro': [class_metrics['f1_score_macro']],
                'cohen_kappa': [class_metrics['cohen_kappa']],
                'weighted_clinical_error': [class_metrics['weighted_clinical_error']]
            }
            pd.DataFrame(class_metrics_data).to_csv(class_metrics_path, index=False)

            cm_path = output_dir / 'prediction_confusion_matrix.csv'
            pd.DataFrame(
                class_metrics['confusion_matrix'],
                index=['normal', 'mild', 'moderate', 'severe'],
                columns=['normal', 'mild', 'moderate', 'severe']
            ).to_csv(cm_path)

            print("\n" + "=" * 60)
            print("Prediction Complete - Classification Files Saved")
            print("=" * 60)
            print(f"{'Classification metrics':<30}: {class_metrics_path}")
            print(f"{'Confusion matrix':<30}: {cm_path}")
            print("=" * 60)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("Predictions Saved")
    print("=" * 60)
    print(f"{'Predictions file':<30}: {output_path}")
    print("=" * 60)


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
        '--config',
        type=str,
        default=None,
        help='Path to JSON config file with hyperparameters (e.g., best_hyperparameters.json from Optuna)'
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
        '--min-child-weight',
        type=int,
        default=1,
        help='Minimum sum of instance weight needed in a child'
    )
    train_parser.add_argument(
        '--subsample',
        type=float,
        default=1.0,
        help='Subsample ratio of training instances'
    )
    train_parser.add_argument(
        '--colsample-bytree',
        type=float,
        default=1.0,
        help='Subsample ratio of columns when constructing each tree'
    )
    train_parser.add_argument(
        '--gamma',
        type=float,
        default=0.0,
        help='Minimum loss reduction required to make a split'
    )
    train_parser.add_argument(
        '--reg-alpha',
        type=float,
        default=0.0,
        help='L1 regularization term on weights'
    )
    train_parser.add_argument(
        '--reg-lambda',
        type=float,
        default=1.0,
        help='L2 regularization term on weights'
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
        if args.config:
            print(f"Loading hyperparameters from {args.config}")
            config = load_config_from_json(Path(args.config))

            train_model(
                csv_path=Path(args.csv),
                output_dir=Path(args.output),
                poly_degree=args.poly_degree,
                spline_smoothing=args.spline_smoothing,
                n_estimators=config.get('n_estimators', args.n_estimators),
                max_depth=config.get('max_depth', args.max_depth),
                learning_rate=config.get('learning_rate', args.learning_rate),
                min_child_weight=config.get('min_child_weight', args.min_child_weight),
                subsample=config.get('subsample', args.subsample),
                colsample_bytree=config.get('colsample_bytree', args.colsample_bytree),
                gamma=config.get('gamma', args.gamma),
                reg_alpha=config.get('reg_alpha', args.reg_alpha),
                reg_lambda=config.get('reg_lambda', args.reg_lambda),
                validation_split=args.val_split
            )
        else:
            train_model(
                csv_path=Path(args.csv),
                output_dir=Path(args.output),
                poly_degree=args.poly_degree,
                spline_smoothing=args.spline_smoothing,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                min_child_weight=args.min_child_weight,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                gamma=args.gamma,
                reg_alpha=args.reg_alpha,
                reg_lambda=args.reg_lambda,
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
