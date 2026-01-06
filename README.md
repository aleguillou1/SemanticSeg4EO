# SemanticSeg4EO

**A Unified Framework for Semantic Segmentation of Earth Observation Imagery**

SemanticSeg4EO is a comprehensive framework for semantic segmentation of satellite imagery, supporting both binary and multi-class segmentation through a unified codebase. The system integrates advanced deep learning architectures specifically adapted for remote sensing applications, with emphasis on methodological transparency, reproducibility, and experimental flexibility.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Patch Extraction](#patch-extraction)
- [Training System](#training-system)
- [Inference on Large Images](#inference-on-large-images)
- [Architecture Support](#architecture-support)
- [Output Format](#output-format)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

## Overview

SemanticSeg4EO provides a unified pipeline for Earth Observation (EO) data segmentation, from data preparation to large-scale inference. The framework combines robust preprocessing, advanced training techniques, and seamless patch-based prediction, making it suitable for both research and production applications in land-cover mapping, environmental monitoring, and change detection.

## Key Features

### Unified Architecture
- Single codebase for both binary and multi-class segmentation
- Automatic mode detection based on configuration
- Consistent interface across all workflows

### Advanced Training Capabilities
- K-Fold Cross-Validation with comprehensive statistics
- Multi-channel data augmentation tailored for satellite imagery
- Class weighting for imbalanced datasets
- Early stopping and model checkpointing
- Percentile-based normalization (99th percentile robust normalization)

### Flexible Model Support
- Custom U-Net variants with dropout regularization
- Segmentation Models PyTorch (SMP) integration:
  - UNet, UNet++, DeepLabV3, DeepLabV3+
  - FPN, PSPNet, MANet, PAN, LinkNet
- TorchVision models support
- Configurable encoders (ResNet, EfficientNet, etc.)

### Large-Scale Inference
- Patch-based prediction with seamless reconstruction
- Weighted blending to reduce border artifacts
- Geospatial metadata preservation
- Confidence map generation

### Data Preparation
- Automatic patch extraction using shapefile grids
- Train/validation/test splitting with reproducibility
- Multi-channel support (including Sentinel-2 with 10+ bands)

## Installation

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.10 (with CUDA for GPU acceleration)
- GPU recommended for training and large-scale inference

### Installation Steps
```bash
# Clone repository
git clone https://github.com/your-username/SemanticSeg4EO.git
cd SemanticSeg4EO

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```text
torch>=1.10.0
torchvision>=0.11.0
segmentation-models-pytorch>=0.3.0
rasterio>=1.2.0
geopandas>=0.10.0
tifffile>=2021.7.2
numpy>=1.21.0
matplotlib>=3.4.0
scipy>=1.7.0
scikit-learn>=0.24.0
tqdm>=4.62.0
opencv-python>=4.5.0
```

## Quick Start

### 1. Prepare Data Structure
```
dataset_root/
├── Patch/
│   ├── train/
│   │   ├── images/
│   │   │   ├── patch_001.tif
│   │   │   └── ...
│   │   └── labels/
│   │       ├── patch_001.tif
│   │       └── ...
│   ├── validation/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
```

### 2. Train a Model
```bash
# Binary segmentation
python main.py --mode binary --dataset_root /path/to/data --model unet++

# Multi-class segmentation (6 classes)
python main.py --mode multiclass --classes 6 --dataset_root /path/to/data --model deeplabv3+
```

### 3. Predict on Large Image
```bash
python Predict_large_image.py --model trained_models/model_final.pth \
                             --input large_image.tif \
                             --output prediction.tif
```

## Dataset Preparation

### Data Format Requirements

- **Images**: Multi-band GeoTIFF files (e.g., Sentinel-2 with 10+ bands)
- **Labels**: Single-band GeoTIFF masks
  - **Binary**: 0 (background) and 1 (foreground)
  - **Multi-class**: Integers from 0 to N-1 (where N = number of classes)
- **Spatial alignment**: Images and masks must have identical georeferencing

### Dataset Structure

The system expects the following directory structure:
```
dataset_root/
└── Patch/
    ├── train/
    │   ├── images/    # Training images
    │   └── labels/    # Training masks
    ├── validation/
    │   ├── images/    # Validation images
    │   └── labels/    # Validation masks
    └── test/
        ├── images/    # Test images
        └── labels/    # Test masks
```

## Patch Extraction

For large satellite scenes, use the patch extraction module to create training-ready datasets:

### Extraction Command
```bash
python Patch_extraction.py extract \
    --image /path/to/satellite_image.tif \
    --label /path/to/ground_truth.tif \
    --grid /path/to/grid_shapefile.shp \
    --output /path/to/output_dataset \
    --patch_size 224 \
    --image_channels 10 \
    --train_ratio 0.75 \
    --val_ratio 0.15 \
    --test_ratio 0.10 \
    --save_metadata
```

### Key Extraction Features

- Grid-based cropping using shapefile polygons
- Automatic resizing to specified patch size
- Dataset splitting with reproducible randomization
- Geospatial metadata preservation
- Optional metadata JSON for traceability

### Visualization of Extracted Patches
```bash
python Patch_extraction.py visualize \
    --output /path/to/output_dataset \
    --split train \
    --sample_index 0
```

## Training System

### Unified Training Interface

The system provides a single entry point (`main.py`) for both segmentation modes:
```bash
python main.py --mode [binary|multiclass] [OPTIONS]
```

### Basic Training Examples

#### Standard Training (Fixed Split)
```bash
# Binary segmentation with data augmentation
python main.py --mode binary \
               --dataset_root /path/to/data \
               --model unet-dropout \
               --data_augmentation \
               --use_class_weights \
               --epochs 100 \
               --batch_size 4

# Multi-class segmentation with pretrained encoder
python main.py --mode multiclass \
               --classes 6 \
               --dataset_root /path/to/data \
               --model deeplabv3+ \
               --encoder_name resnet50 \
               --pretrained \
               --epochs 150 \
               --batch_size 8
```

#### Cross-Validation Training
```bash
# 5-fold cross-validation for robust evaluation
python main.py --mode multiclass \
               --classes 5 \
               --dataset_root /path/to/data \
               --model unet++ \
               --val_strategy kfold \
               --n_splits 5 \
               --data_augmentation \
               --use_class_weights
```

### Available Training Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode` | Segmentation mode: `binary` or `multiclass` | `binary` |
| `--dataset_root` | Path to dataset root directory | Required |
| `--model` | Model architecture name | Required |
| `--classes` | Number of classes (for multiclass) | 2 |
| `--val_strategy` | Validation strategy: `split` or `kfold` | `split` |
| `--epochs` | Number of training epochs | 100 |
| `--batch_size` | Batch size | 4 |
| `--learning_rate` | Learning rate | 1e-3 |
| `--encoder_name` | Encoder backbone name | `resnet34` |
| `--pretrained` | Use pretrained encoder weights | False |
| `--data_augmentation` | Enable multi-channel data augmentation | False |
| `--use_class_weights` | Apply class weights for imbalance | False |
| `--n_splits` | Number of folds for cross-validation | 5 |

### Training Output

During training, the system generates:
```
trained_models/
├── model_best_loss.pth          # Best validation loss checkpoint
├── model_best_iou.pth           # Best IoU checkpoint
├── model_best_combined.pth      # Combined best metrics
├── model_final_model.pth        # Final trained model
├── model_metrics.json           # Detailed metrics and history
└── model_training_plot.png      # Training visualization
```

## Inference on Large Images

The `Predict_large_image.py` script handles prediction on arbitrarily large satellite scenes:

### Basic Prediction
```bash
python Predict_large_image.py --model /path/to/model.pth \
                             --input /path/to/large_image.tif \
                             --output /path/to/prediction.tif
```

### Advanced Prediction Options
```bash
# Multi-class with custom parameters
python Predict_large_image.py \
    --model /path/to/model.pth \
    --input large_image.tif \
    --output prediction.tif \
    --num_classes 6 \
    --patch_size 512 \
    --overlap 128 \
    --save_confidence \
    --device cuda
```

### Prediction Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Path to trained model (.pth file) | Required |
| `--input` | Input satellite image | Required |
| `--output` | Output segmentation map | Required |
| `--patch_size` | Size of prediction patches | 512 |
| `--overlap` | Overlap between patches | 128 |
| `--num_classes` | Number of output classes | Auto-detected |
| `--threshold` | Confidence threshold (binary) | 0.5 |
| `--save_confidence` | Save confidence map | False |
| `--device` | Computation device | `cuda` |

### Seamless Reconstruction

The predictor uses:
- Weighted blending to eliminate border artifacts
- Automatic patch tiling with configurable overlap
- Geospatial metadata preservation
- Nodata value handling from source images

## Architecture Support

### Available Models

Run the following to see all available models:
```bash
python -c "from model_training import get_available_models; print(get_available_models())"
```

### Model Categories

#### 1. Custom Models
- `unet-dropout`: Custom U-Net with dropout regularization

#### 2. SMP Models (requires segmentation-models-pytorch)
- **Encoder-Decoder**: UNet, UNet++, MANet, Linknet, PAN
- **Pyramid Networks**: FPN, PSPNet
- **DeepLab Family**: DeepLabV3, DeepLabV3+

#### 3. TorchVision Models
- **DeepLabV3**: With ResNet50 backbone

### Supported Encoders

- **ResNet**: 18, 34, 50, 101, 152
- **EfficientNet**: b0-b7
- **MobileNet**: v2, v3
- **DenseNet**: 121, 169, 201, 264
- **VGG**: 11, 13, 16, 19
- And more via SMP

## Output Format

### Model Checkpoints

Trained models are saved with comprehensive metadata:
```python
{
    'model_state_dict': model_weights,
    'metadata': {
        'model_name': 'unet++',
        'mode': 'multiclass',
        'in_channels': 10,
        'num_classes': 6,
        'input_size': [224, 224],
        'normalization': 'percentile_99',
        'encoder_name': 'resnet34',
        'pretrained': True,
        'training_params': {
            'epochs': 100,
            'batch_size': 4,
            'learning_rate': 0.001,
            'best_val_loss': 0.1234,
            'best_val_iou': 0.7890
        },
        'performance_metrics': {...}
    }
}
```

### Prediction Outputs

- **Segmentation map**: GeoTIFF with class labels
- **Confidence map** (optional): GeoTIFF with prediction confidence
- **Statistics report**: Console output with class distribution and confidence metrics

## Examples

### Example 1: Land Cover Classification (Multi-class)
```bash
# Extract patches from large scenes
python Patch_extraction.py extract \
    --image sentinel2_scene.tif \
    --label landcover_labels.tif \
    --grid grid_polygons.shp \
    --output ./landcover_dataset \
    --patch_size 256 \
    --image_channels 10 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15

# Train with cross-validation
python main.py --mode multiclass \
               --classes 6 \
               --dataset_root ./landcover_dataset \
               --model deeplabv3+ \
               --encoder_name efficientnet-b4 \
               --pretrained \
               --val_strategy kfold \
               --n_splits 5 \
               --data_augmentation \
               --use_class_weights \
               --epochs 200 \
               --save_dir ./landcover_models

# Predict on new large scene
python Predict_large_image.py \
    --model ./landcover_models/model_final_model.pth \
    --input new_sentinel2_scene.tif \
    --output landcover_prediction.tif \
    --save_confidence
```

### Example 2: Water Body Detection (Binary)
```bash
# Train binary segmentation
python main.py --mode binary \
               --dataset_root ./water_dataset \
               --model unet++ \
               --encoder_name resnet34 \
               --pretrained \
               --in_channels 10 \
               --data_augmentation \
               --use_class_weights \
               --epochs 150 \
               --learning_rate 0.0005

# Predict with custom threshold
python Predict_large_image.py \
    --model ./water_models/model_final_model.pth \
    --input sentinel2_water_scene.tif \
    --output water_mask.tif \
    --threshold 0.3 \
    --patch_size 224 \
    --overlap 64
```

## Best Practices

### Data Preparation
- Normalize images using the 99th percentile method (already implemented)
- Ensure class balance or use `--use_class_weights` for imbalanced datasets
- Use data augmentation (`--data_augmentation`) for small datasets
- Validate spatial alignment between images and masks

### Training Configuration
- Start with pretrained encoders for faster convergence
- Use cross-validation (`--val_strategy kfold`) for reliable performance estimation
- Adjust batch size based on GPU memory (typically 4-16 for 224-512px patches)
- Monitor multiple metrics: Loss, IoU, and F1-score

### Inference Settings
- Set appropriate overlap (25-50% of patch size) to avoid border artifacts
- Generate confidence maps (`--save_confidence`) for uncertainty analysis
- Process large images in chunks if memory is limited
- Verify geospatial alignment of output predictions

### Performance Optimization
- Use GPU acceleration for both training and inference
- Adjust patch size based on GPU memory (256-512px recommended)
- Enable mixed precision training for faster training (modify `model_training.py`)
- Use data loaders with pinned memory for faster data transfer

## Troubleshooting

### Common Issues

#### 1. "No images found" error
- **Cause**: Incorrect dataset structure or file extensions
- **Solution**: Verify directory structure and ensure files have `.tif` or `.tiff` extensions

#### 2. CUDA out of memory
- **Cause**: Batch size or patch size too large
- **Solution**: Reduce `--batch_size` or `--patch_size`

#### 3. Poor prediction quality at patch borders
- **Cause**: Insufficient overlap between patches
- **Solution**: Increase `--overlap` parameter (recommended: 25-50% of patch size)

#### 4. Model fails to load
- **Cause**: Mismatch in model parameters or architecture
- **Solution**: Ensure `--in_channels` and `--num_classes` match training configuration

#### 5. Slow inference speed
- **Cause**: Large patch size or CPU inference
- **Solution**: Reduce patch size, use GPU (`--device cuda`), or enable tiling

### Debug Mode

For detailed debugging, add error tracebacks:
```python
# In model_training.py or Predict_large_image.py
import traceback
try:
    # Your code here
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions, collaborations, or technical support:

**Adrien Leguillou**  
Research Engineer at LETG 
Email: adrien.leguillou@univ-brest.fr  


## Acknowledgments

This framework builds upon several open-source projects:
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [PyTorch](https://pytorch.org/)
- [Rasterio](https://rasterio.readthedocs.io/)
- [GDAL](https://gdal.org/)

Special thanks to the remote sensing community for datasets and methodologies that inspired this work.
