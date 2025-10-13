"""
Generate data index CSV file from spinal AI dataset annotations.

This script processes spinal AI dataset annotations including:
- JSON annotations from zip files (bounding boxes, segmentation, transcriptions)
- Cobb angle measurements from text files (three angles per image)

The output is a comprehensive CSV file containing all annotations merged by image filename.
"""

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Any
import csv


def extract_json_from_zip(zip_path: Path, cache_dir: Path) -> Dict[str, Any]:
    """
    Extract and parse JSON file from zip archive.

    Args:
        zip_path: Path to the zip file
        cache_dir: Directory to extract files to

    Returns:
        Parsed JSON data as dictionary
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        json_files = [f for f in zip_ref.namelist() if f.endswith('.json')]
        if not json_files:
            raise ValueError(f"No JSON file found in {zip_path}")

        json_filename = json_files[0]
        extracted_path = cache_dir / json_filename

        zip_ref.extract(json_filename, cache_dir)

        with open(extracted_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def parse_cobb_angles(txt_path: Path) -> Dict[str, List[float]]:
    """
    Parse Cobb angle measurements from text file.

    Args:
        txt_path: Path to the Cobb angles text file

    Returns:
        Dictionary mapping filename to list of three angles [angle1, angle2, angle3]
    """
    cobb_data = {}

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) != 4:
                continue

            filename = parts[0]
            angles = [float(parts[1]), float(parts[2]), float(parts[3])]
            cobb_data[filename] = angles

    return cobb_data


def build_image_path_map(data_dir: Path) -> Dict[str, str]:
    """
    Build a mapping from filename to its subdirectory path.

    Images are distributed across 5 subset directories:
    - Spinal-AI2024-subset1: 000001-004000
    - Spinal-AI2024-subset2: 004001-008000
    - Spinal-AI2024-subset3: 008001-012000
    - Spinal-AI2024-subset4: 012001-016000
    - Spinal-AI2024-subset5: 016001-020000

    Args:
        data_dir: Root data directory containing subset subdirectories

    Returns:
        Dictionary mapping filename to relative path from project root
    """
    image_path_map = {}

    subset_dirs = [
        'Spinal-AI2024-subset1',
        'Spinal-AI2024-subset2',
        'Spinal-AI2024-subset3',
        'Spinal-AI2024-subset4',
        'Spinal-AI2024-subset5'
    ]

    for subset_dir in subset_dirs:
        subset_path = data_dir / subset_dir
        if not subset_path.exists():
            continue

        for img_file in subset_path.glob('*.jpg'):
            filename = img_file.name
            relative_path = f"data/{subset_dir}/{filename}"
            image_path_map[filename] = relative_path

    return image_path_map


def process_annotations(json_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Process annotation data and organize by image.

    Args:
        json_data: Parsed JSON annotation data

    Returns:
        Dictionary mapping image_id to aggregated annotation information
    """
    images_by_id = {img['id']: img for img in json_data['images']}

    annotations_by_image = {}

    for ann in json_data['annotations']:
        image_id = ann['image_id']

        if image_id not in annotations_by_image:
            image_info = images_by_id[image_id]
            annotations_by_image[image_id] = {
                'filename': image_info['file_name'],
                'width': image_info['width'],
                'height': image_info['height'],
                'annotations': []
            }

        annotations_by_image[image_id]['annotations'].append({
            'bbox': ann['bbox'],
            'area': ann['area'],
            'transcription': ann.get('transcription', ''),
            'score': ann.get('score', 1.0)
        })

    return annotations_by_image


def generate_csv(
    train_annotations: Dict[str, Dict[str, Any]],
    test_annotations: Dict[str, Dict[str, Any]],
    train_cobb: Dict[str, List[float]],
    test_cobb: Dict[str, List[float]],
    output_path: Path,
    image_path_map: Dict[str, str]
) -> None:
    """
    Generate comprehensive CSV file with all data.

    Args:
        train_annotations: Processed training annotations
        test_annotations: Processed test annotations
        train_cobb: Training Cobb angles
        test_cobb: Test Cobb angles
        output_path: Path to output CSV file
        image_path_map: Mapping from filename to relative path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_data = []

    for dataset_name, annotations, cobb_data in [
        ('train', train_annotations, train_cobb),
        ('test', test_annotations, test_cobb)
    ]:
        for img_data in annotations.values():
            filename = img_data['filename']

            angles = cobb_data.get(filename, [0.0, 0.0, 0.0])

            relative_path = image_path_map.get(filename, f"data/{filename}")

            num_annotations = len(img_data['annotations'])
            total_area = sum(ann['area'] for ann in img_data['annotations'])
            avg_score = (
                sum(ann['score'] for ann in img_data['annotations']) / num_annotations
                if num_annotations > 0 else 0.0
            )

            transcriptions = ';'.join(
                ann['transcription'] for ann in img_data['annotations']
                if ann['transcription']
            )

            bboxes = ';'.join(
                f"{ann['bbox'][0]},{ann['bbox'][1]},{ann['bbox'][2]},{ann['bbox'][3]}"
                for ann in img_data['annotations']
            )

            row = {
                'filename': filename,
                'relative_path': relative_path,
                'dataset': dataset_name,
                'width': img_data['width'],
                'height': img_data['height'],
                'cobb_angle_1': angles[0],
                'cobb_angle_2': angles[1],
                'cobb_angle_3': angles[2],
                'num_annotations': num_annotations,
                'total_annotation_area': total_area,
                'avg_annotation_score': avg_score,
                'transcriptions': transcriptions,
                'bboxes': bboxes
            }

            all_data.append(row)

    all_data.sort(key=lambda x: x['filename'])

    fieldnames = [
        'filename',
        'relative_path',
        'dataset',
        'width',
        'height',
        'cobb_angle_1',
        'cobb_angle_2',
        'cobb_angle_3',
        'num_annotations',
        'total_annotation_area',
        'avg_annotation_score',
        'transcriptions',
        'bboxes'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)


def main() -> None:
    """
    Main entry point for the index generation script.

    Processes all annotation files and generates a comprehensive data.csv file
    in the data_index directory.
    """
    root_dir = Path(__file__).parent.parent.parent
    data_dir = root_dir / 'data'
    cache_dir = root_dir / 'cache'
    output_dir = root_dir / 'data_index'

    print("Starting data index generation...")

    print("Extracting test annotations...")
    test_json = extract_json_from_zip(
        data_dir / 'Spinal_AI2024_test_annotation.zip',
        cache_dir
    )

    print("Extracting train annotations...")
    train_json = extract_json_from_zip(
        data_dir / 'Spinal_AI2024_train__annotation.zip',
        cache_dir
    )

    print("Parsing Cobb angles...")
    test_cobb = parse_cobb_angles(data_dir / 'Cobb_spinal-AI2024-test_gt.txt')
    train_cobb = parse_cobb_angles(data_dir / 'Cobb_spinal-AI2024-train_gt.txt')

    print("Processing annotations...")
    test_annotations = process_annotations(test_json)
    train_annotations = process_annotations(train_json)

    print("Building image path map...")
    image_path_map = build_image_path_map(data_dir)

    print("Generating CSV file...")
    output_path = output_dir / 'data.csv'
    generate_csv(
        train_annotations,
        test_annotations,
        train_cobb,
        test_cobb,
        output_path,
        image_path_map
    )

    print(f"Successfully generated {output_path}")
    print(f"Total images: {len(train_annotations) + len(test_annotations)}")
    print(f"- Train images: {len(train_annotations)}")
    print(f"- Test images: {len(test_annotations)}")


if __name__ == '__main__':
    main()
