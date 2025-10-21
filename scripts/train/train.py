"""
Training script for YOLOv11 object detection on spinal bone dataset.

This script trains a complete YOLOv11 model (not transfer learning) for detecting
spinal bones in medical images. It processes the data index CSV, prepares the dataset
in YOLO format, and trains the model with appropriate configurations.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import shutil

from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]


def load_data_index(csv_path: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Load and parse the data index CSV file.

    Args:
        csv_path: Path to the data.csv file

    Returns:
        Tuple of (train_data, test_data) where each is a list of dictionaries
    """
    train_data = []
    test_data = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['dataset'] == 'train':
                train_data.append(row)
            else:
                test_data.append(row)

    return train_data, test_data


def parse_bbox(bbox_str: str) -> List[Tuple[int, int, int, int]]:
    """
    Parse bounding box string into list of bboxes.

    Args:
        bbox_str: Semicolon-separated string of bboxes in format "x,y,w,h;x,y,w,h;..."

    Returns:
        List of (x, y, w, h) tuples
    """
    if not bbox_str:
        return []

    bboxes = []
    for bbox in bbox_str.split(';'):
        parts = bbox.split(',')
        if len(parts) == 4:
            x, y, w, h = map(int, parts)
            bboxes.append((x, y, w, h))

    return bboxes


def convert_to_yolo_format(
    bbox: Tuple[int, int, int, int], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert bbox from (x, y, w, h) to YOLO format (center_x, center_y, w, h) normalized.

    Args:
        bbox: Bounding box in format (x, y, w, h)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (center_x, center_y, width, height) normalized to [0, 1]
    """
    x, y, w, h = bbox

    center_x = (x + w / 2) / img_width
    center_y = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height

    return (center_x, center_y, width, height)


def prepare_yolo_dataset(
    train_data: List[Dict],
    test_data: List[Dict],
    root_dir: Path,
    output_dir: Path
) -> Path:
    """
    Prepare dataset in YOLO format with labels and data.yaml configuration.

    Args:
        train_data: List of training data entries
        test_data: List of test data entries
        root_dir: Project root directory
        output_dir: Directory to save YOLO formatted dataset

    Returns:
        Path to the generated data.yaml file
    """
    dataset_dir = output_dir / 'yolo_dataset'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_images_dir = dataset_dir / 'images' / 'train'
    test_images_dir = dataset_dir / 'images' / 'val'
    train_labels_dir = dataset_dir / 'labels' / 'train'
    test_labels_dir = dataset_dir / 'labels' / 'val'

    train_images_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    for data_list, images_dir, labels_dir in [
        (train_data, train_images_dir, train_labels_dir),
        (test_data, test_images_dir, test_labels_dir)
    ]:
        for entry in data_list:
            filename = entry['filename']
            img_path = root_dir / entry['relative_path']

            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue

            target_img_path = images_dir / filename
            if not target_img_path.exists():
                shutil.copy2(img_path, target_img_path)

            label_path = labels_dir / filename.replace('.jpg', '.txt')

            bboxes = parse_bbox(entry['bboxes'])
            img_width = int(entry['width'])
            img_height = int(entry['height'])

            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    center_x, center_y, width, height = convert_to_yolo_format(
                        bbox, img_width, img_height
                    )
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    data_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'spinal_bone'
        }
    }

    yaml_path = dataset_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    return yaml_path


def train_yolo11(
    data_yaml_path: Path,
    output_dir: Path,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = '0',
    workers: int = 8,
    patience: int = 50,
    save_period: int = 10
) -> None:
    """
    Train YOLOv11 model from scratch on spinal bone detection dataset.

    The model will be saved to output_dir/spinal_bone_detection/weights/.
    Note: Model weights are only saved after completing at least one epoch.
    Checkpoints are saved every save_period epochs, and the best model is
    automatically saved based on validation metrics.

    Args:
        data_yaml_path: Path to the data.yaml configuration file
        output_dir: Directory to save trained models
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size for training
        device: Device to train on ('0' for GPU, 'cpu' for CPU)
        workers: Number of dataloader workers
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
    """
    model = YOLO('yolo11n.yaml')

    print("\n" + "=" * 60)
    print("Starting YOLOv11 Training")
    print("=" * 60)
    print(f"Data config: {data_yaml_path}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print("=" * 60 + "\n")

    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        save_period=save_period,
        project=str(output_dir),
        name='spinal_bone_detection',
        exist_ok=True,
        pretrained=False,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model saved to: {output_dir / 'spinal_bone_detection' / 'weights' / 'best.pt'}")
    print(f"Last model saved to: {output_dir / 'spinal_bone_detection' / 'weights' / 'last.pt'}")
    print("=" * 60 + "\n")


def main() -> None:
    """
    Main entry point for the training script.

    Loads data index, prepares YOLO dataset, and trains YOLOv11 model.
    """
    root_dir = Path(__file__).parent.parent.parent
    data_csv = root_dir / 'data_index' / 'data.csv'
    cache_dir = root_dir / 'cache'
    models_dir = root_dir / 'models'

    if not data_csv.exists():
        raise FileNotFoundError(
            f"Data index not found at {data_csv}. "
            "Please run 'uv run index_gen' first to generate the data index."
        )

    print("Loading data index...")
    train_data, test_data = load_data_index(data_csv)
    print(f"Loaded {len(train_data)} training images and {len(test_data)} test images")

    print("\nPreparing YOLO dataset...")
    data_yaml_path = prepare_yolo_dataset(train_data, test_data, root_dir, cache_dir)
    print(f"Dataset prepared at: {data_yaml_path}")

    print("\nStarting training...")
    train_yolo11(
        data_yaml_path=data_yaml_path,
        output_dir=models_dir,
        epochs=100,
        imgsz=640,
        batch=16,
        device='0'
    )


if __name__ == '__main__':
    main()
