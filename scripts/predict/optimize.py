"""
Optuna hyperparameter optimization for XGBoost Cobb angle predictor.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from .feature_extractor import CurveFeatureExtractor
from .predict import (
    load_data_from_csv,
    extract_features_from_dataframe,
    prepare_targets
)


def objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5
) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial: Optuna trial object
        X: Feature matrix with shape (n_samples, n_features)
        y: Target matrix with shape (n_samples, 3)
        n_splits: Number of cross-validation splits

    Returns:
        Mean cross-validation MAE across all three Cobb angles
    """
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 3, 12)
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3, log=True)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    gamma = trial.suggest_float("gamma", 0.0, 5.0)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)

    base_model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=-1
    )

    model = MultiOutputRegressor(base_model)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mae_scores = []
    for train_idx, val_idx in kfold.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_val_fold)

        mae_fold = mean_absolute_error(y_val_fold, y_pred_fold)
        mae_scores.append(mae_fold)

    mean_mae = float(np.mean(mae_scores))

    return mean_mae


def optimize_hyperparameters(
    csv_path: Path,
    output_dir: Path,
    n_trials: int = 100,
    poly_degree: int = 7,
    spline_smoothing: float = 1.0,
    n_splits: int = 5,
    timeout: int | None = None
) -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using Optuna.

    Args:
        csv_path: Path to training data CSV
        output_dir: Directory to save optimization results
        n_trials: Number of optimization trials
        poly_degree: Polynomial degree for curve fitting
        spline_smoothing: Smoothing factor for spline interpolation
        n_splits: Number of cross-validation splits
        timeout: Maximum time in seconds for optimization (None for no limit)

    Returns:
        Dictionary containing best hyperparameters and trial statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_data_from_csv(csv_path)

    print("Extracting features...")
    feature_extractor = CurveFeatureExtractor(
        poly_degree=poly_degree,
        spline_smoothing=spline_smoothing
    )

    X, valid_indices = extract_features_from_dataframe(df, feature_extractor)
    y = prepare_targets(df, valid_indices)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")

    print("\n" + "=" * 60)
    print("Starting Optuna Hyperparameter Optimization")
    print("=" * 60)
    print(f"Number of trials: {n_trials}")
    print(f"Cross-validation splits: {n_splits}")
    if timeout:
        print(f"Timeout: {timeout} seconds")
    print("=" * 60 + "\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5
        )
    )

    study.optimize(
        lambda trial: objective(trial, X, y, n_splits),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial MAE: {study.best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key:<20}: {value}")
    print("=" * 60 + "\n")

    best_params_path = output_dir / 'best_hyperparameters.json'
    with open(best_params_path, 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)
    print(f"Best hyperparameters saved to {best_params_path}")

    trials_df = study.trials_dataframe()
    trials_path = output_dir / 'optimization_trials.csv'
    trials_df.to_csv(trials_path, index=False)
    print(f"All trials saved to {trials_path}")

    visualization_dir = output_dir / 'visualizations'
    visualization_dir.mkdir(exist_ok=True)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(str(visualization_dir / 'optimization_history.html'))

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(str(visualization_dir / 'param_importances.html'))

    fig = optuna.visualization.plot_slice(study)
    fig.write_html(str(visualization_dir / 'param_slice.html'))

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_html(str(visualization_dir / 'parallel_coordinate.html'))

    print(f"Visualizations saved to {visualization_dir}")

    results = {
        'best_params': study.best_trial.params,
        'best_value': study.best_trial.value,
        'n_trials': len(study.trials),
        'poly_degree': poly_degree,
        'spline_smoothing': spline_smoothing
    }

    return results


def main():
    """
    Main entry point for hyperparameter optimization.
    """
    parser = argparse.ArgumentParser(
        description='Optimize XGBoost hyperparameters for Cobb angle prediction'
    )

    parser.add_argument(
        '--csv',
        type=str,
        default='data_index/data.csv',
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/optuna_results',
        help='Output directory for optimization results'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of optimization trials'
    )
    parser.add_argument(
        '--poly-degree',
        type=int,
        default=7,
        help='Polynomial degree for curve fitting'
    )
    parser.add_argument(
        '--spline-smoothing',
        type=float,
        default=1.0,
        help='Smoothing factor for spline interpolation'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Number of cross-validation splits'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Maximum time in seconds for optimization'
    )

    args = parser.parse_args()

    results = optimize_hyperparameters(
        csv_path=Path(args.csv),
        output_dir=Path(args.output),
        n_trials=args.n_trials,
        poly_degree=args.poly_degree,
        spline_smoothing=args.spline_smoothing,
        n_splits=args.n_splits,
        timeout=args.timeout
    )

    print("\nOptimization results:")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
