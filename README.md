# Spinal Medical Image Analysis

A machine learning project for spinal bone detection and Cobb angle prediction from medical X-ray images using YOLOv11 and XGBoost.

## Overview

This project provides a complete pipeline for analyzing spinal medical images:

1. **Spinal Bone Detection**: YOLOv11-based object detection to locate spinal vertebrae
2. **Cobb Angle Prediction**: Curve fitting and XGBoost regression to predict Cobb angles for scoliosis assessment

The Cobb angle is a standard measurement used by orthopedic surgeons to assess the severity of spinal curvature in conditions like scoliosis.

## Project Structure

```
Code/
├── data/                                      # Dataset directory
│   ├── Spinal-AI2024-subset1/                 # Images 000001-004000
│   ├── Spinal-AI2024-subset2/                 # Images 004001-008000
│   ├── Spinal-AI2024-subset3/                 # Images 008001-012000
│   ├── Spinal-AI2024-subset4/                 # Images 012001-016000
│   ├── Spinal-AI2024-subset5/                 # Images 016001-020000
│   ├── Spinal_AI2024_train__annotation.zip    # Training annotations
│   ├── Spinal_AI2024_test_annotation.zip      # Test annotations
│   ├── Cobb_spinal-AI2024-train_gt.txt        # Training Cobb angles
│   └── Cobb_spinal-AI2024-test_gt.txt         # Test Cobb angles
├── data_index/                                # Generated data index
│   └── data.csv                               # Comprehensive dataset index
├── scripts/                                   # Source code
│   ├── pre/                                   # Preprocessing scripts
│   │   └── index_gen.py                       # Generate data index from annotations
│   ├── train/                                 # Training scripts
│   │   └── train.py                           # Train YOLOv11 detection model
│   └── predict/                               # Prediction scripts
│       ├── feature_extractor.py               # Extract curve features from bboxes
│       ├── model.py                           # XGBoost Cobb angle predictor
│       └── predict.py                         # Main prediction pipeline
├── models/                                    # Saved trained models
├── cache/                                     # Cached data and temporary files
├── pyproject.toml                             # Project configuration
└── README.md                                  # This file
```

## Requirements

- Python 3.13+
- CUDA-enabled GPU (recommended for training)
- UV package manager

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Code
```

2. Install dependencies using UV:
```bash
uv sync
```

This will install all required packages including:
- ultralytics (YOLOv11)
- xgboost
- optuna (hyperparameter optimization)
- plotly (optimization visualizations)
- scipy
- scikit-learn
- pandas
- numpy
- PyTorch (with CUDA 12.8 support)

## Usage

### 1. Generate Data Index

First, create a comprehensive CSV index from the dataset annotations:

```bash
uv run index_gen
```

This processes:
- JSON annotations (bounding boxes, segmentation, transcriptions)
- Cobb angle measurements from text files
- Image metadata

Output: `data_index/data.csv`

### 2. Train Spinal Bone Detection Model

Train a YOLOv11 model to detect spinal vertebrae:

```bash
uv run train
```

Key features:
- Trains YOLOv11n architecture from scratch (not transfer learning)
- Automatically converts annotations to YOLO format
- Saves models to `models/spinal_bone_detection/weights/`
- Training parameters:
  - Epochs: 100 (with early stopping patience=50)
  - Image size: 640x640
  - Batch size: 16
  - Device: GPU 0

Outputs:
- `best.pt`: Best model based on validation metrics
- `last.pt`: Final epoch checkpoint

### 3. Optimize XGBoost Hyperparameters (Optional)

Before training the Cobb angle predictor, you can optimize hyperparameters using Optuna:

```bash
uv run optimize --csv data_index/data.csv --output models/optuna_results
```

Options:
- `--n-trials`: Number of optimization trials (default: 100)
- `--poly-degree`: Polynomial degree for curve fitting (default: 7)
- `--spline-smoothing`: Spline smoothing factor (default: 1.0)
- `--n-splits`: Cross-validation splits (default: 5)
- `--timeout`: Maximum optimization time in seconds (optional)

This will:
- Perform hyperparameter optimization using Optuna with TPE sampler
- Use 5-fold cross-validation to evaluate each trial
- Save best hyperparameters to `models/optuna_results/best_hyperparameters.json`
- Save all trial results to `models/optuna_results/optimization_trials.csv`
- Generate interactive HTML visualizations in `models/optuna_results/visualizations/`:
  - optimization_history.html: Trial values over time
  - param_importances.html: Most influential hyperparameters
  - param_slice.html: Individual parameter effects
  - parallel_coordinate.html: Multi-dimensional parameter relationships

### 4. Train Cobb Angle Predictor

#### Train the Cobb Angle Predictor

**Option 1: Train with default or manual parameters**
```bash
uv run predict train --csv data_index/data.csv --output models/cobb_predictor
```

**Option 2: Train with optimized hyperparameters from Optuna (recommended after optimization)**
```bash
uv run predict train --csv data_index/data.csv --output models/cobb_predictor_optimized \
  --config models/optuna_results/best_hyperparameters.json
```

Options:
- `--config`: Path to JSON config file with hyperparameters (e.g., `best_hyperparameters.json` from Optuna)
- `--poly-degree`: Polynomial degree for curve fitting (default: 7)
- `--spline-smoothing`: Spline smoothing factor (default: 1.0)
- `--n-estimators`: XGBoost estimator count (default: 100)
- `--max-depth`: Maximum tree depth (default: 6)
- `--learning-rate`: Learning rate (default: 0.1)
- `--min-child-weight`: Minimum sum of instance weight (default: 1)
- `--subsample`: Subsample ratio (default: 1.0)
- `--colsample-bytree`: Column subsample ratio (default: 1.0)
- `--gamma`: Minimum loss reduction (default: 0.0)
- `--reg-alpha`: L1 regularization (default: 0.0)
- `--reg-lambda`: L2 regularization (default: 1.0)
- `--val-split`: Validation split ratio (default: 0.2)

You can use the optimized hyperparameters from step 3 by specifying them here.

#### Run Predictions

```bash
uv run predict predict --model models/cobb_predictor/cobb_angle_predictor.pkl --csv data_index/data.csv --output predictions.csv
```

The predictor outputs three Cobb angles per image (angle_1, angle_2, angle_3).

## Technical Details

### Data Index Format

The generated `data.csv` contains:
- `filename`: Image filename
- `relative_path`: Path to image file
- `dataset`: 'train' or 'test'
- `width`, `height`: Image dimensions
- `cobb_angle_1`, `cobb_angle_2`, `cobb_angle_3`: Ground truth angles
- `num_annotations`: Number of bounding boxes
- `bboxes`: Semicolon-separated bounding boxes (x,y,w,h format)
- `transcriptions`: Vertebra labels
- `total_annotation_area`: Total bbox area
- `avg_annotation_score`: Average annotation confidence

### Feature Extraction

The curve feature extractor:
1. Extracts center points from vertebra bounding boxes
2. Sorts points vertically to form a spinal curve
3. Fits polynomial and spline curves
4. Computes curvature metrics, angles, and geometric features
5. Generates feature vectors for XGBoost prediction

### Model Architecture

- **Detection**: YOLOv11n (nano) for efficient real-time detection
- **Regression**: XGBoost multi-output regressor for three Cobb angles

## Dataset

This project uses the Spinal-AI2024 dataset:
- 20,000 spinal X-ray images (numbered 000001-020000)
- Split into 5 subsets of 4,000 images each
- Annotations include vertebra bounding boxes and Cobb angle measurements
- Training/test split provided in annotations

## License

[Add your license information here]

## Citation

[Add citation information if applicable]

## Acknowledgments

- Ultralytics for YOLOv11 framework
- Spinal-AI2024 dataset contributors
