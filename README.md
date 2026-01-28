# SemanticSeg4EO

**A Unified Framework for Semantic Segmentation of Earth Observation Imagery**

SemanticSeg4EO is a comprehensive framework for semantic segmentation of satellite imagery, supporting both binary and multi-class segmentation through a unified codebase. The system integrates advanced deep learning architectures specifically adapted for remote sensing applications, with emphasis on methodological transparency, reproducibility, and experimental flexibility.

## Table of Contents

- [Overview](#overview)
- [What's New in V2](#whats-new-in-v2)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Patch Extraction](#patch-extraction)
- [Training System](#training-system)
- [Advanced Training Features (V2)](#advanced-training-features-v2)
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

## What's New in V2

Version 2 introduces significant enhancements for improved training performance and flexibility:

### 🔥 New Loss Functions
- **Focal Loss**: Better handling of class imbalance with configurable alpha/gamma
- **Tversky Loss**: Control false positive/negative tradeoff with alpha/beta parameters
- **Combo Loss**: Combined CE + Dice + Focal for maximum flexibility
- **Focal-Dice**: Recommended for severely imbalanced datasets

### 🧊 Transfer Learning Improvements
- **Encoder Freezing**: Freeze pretrained encoder for initial epochs to preserve learned features
- **Gradual Unfreezing**: Automatic unfreezing after specified epochs

### 📈 Learning Rate Enhancements
- **Warmup**: Gradual learning rate increase at training start
- **Multiple Schedulers**: ReduceLROnPlateau, Cosine Annealing, One-Cycle

### ⚡ Performance Optimizations
- **Mixed Precision Training (AMP)**: Faster training with reduced memory usage
- **Per-class IoU Logging**: Detailed metrics for each class during training

### 📊 Better Monitoring
- **CSV Export**: All training metrics saved to CSV for analysis
- **Per-class Visualization**: Training plots show IoU evolution per class
- **Enhanced Checkpoints**: Complete configuration saved with models

### 🗂️ Batch Processing
- **Multi-image Extraction**: Process multiple image-label pairs automatically
- **Pattern Matching**: Automatic file pairing with regex patterns

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
- Segmentation Models PyTorch (SMP) integration: UNet, UNet++, DeepLabV3, DeepLabV3+, FPN, PSPNet, MANet, PAN, LinkNet
- TorchVision models support
- Configurable encoders (ResNet, EfficientNet, etc.)

### Large-Scale Inference
- Patch-based prediction with seamless reconstruction
- Weighted blending to reduce border artifacts
- Geospatial metadata preservation
- Confidence map generation
- Automatic encoder detection from checkpoints (V2)

### Data Preparation
- Automatic patch extraction using shapefile grids
- Train/validation/test splitting with reproducibility
- Multi-channel support (including Sentinel-2 with 10+ bands)
- **Batch mode** for processing multiple images (V2)

## Installation

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.10 (with CUDA for GPU acceleration)
- GPU recommended for training and large-scale inference

### Installation Steps
```bash
# Clone repository
git clone https://github.com/aleguillou1/SemanticSeg4EO.git
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

### 2. Train a Model (V2)
```bash
# Multi-class segmentation with Focal Loss (recommended for imbalanced data)
python main_v2.py --mode multiclass --classes 5 --dataset_root /path/to/data \
    --model unet++ --loss_type focal --use_class_weights

# With encoder freezing and warmup
python main_v2.py --mode multiclass --classes 5 --dataset_root /path/to/data \
    --model unet++ --freeze_encoder --freeze_epochs 5 --warmup_epochs 2

# Binary segmentation
python main_v2.py --mode binary --dataset_root /path/to/data --model unet
```

### 3. Predict on Large Image
```bash
python Predict_large_image_v2.py --model trained_models/model_final.pth \
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

For large satellite scenes, use the patch extraction module to create training-ready datasets.

### Single Image Extraction
```bash
python Patch_extraction_v2.py single \
    --image /path/to/satellite_image.tif \
    --label /path/to/ground_truth.tif \
    --grid /path/to/grid_shapefile.shp \
    --output /path/to/output_dataset \
    --patch_size 224 \
    --image_channels 10 \
    --train_ratio 0.75 \
    --val_ratio 0.15 \
    --test_ratio 0.10
```

### Batch Mode (V2) - Multiple Images
```bash
# Automatically find and process Image_1.tif/Label_1.tif, Image_2.tif/Label_2.tif, etc.
python Patch_extraction_v2.py batch \
    --data_dir /path/to/images_folder \
    --grid /path/to/grid.shp \
    --output /path/to/output \
    --patch_size 224 \
    --image_channels 10 \
    --recursive
```

### Batch Mode File Naming Convention
- Images: `Image_1.tif`, `Image_2.tif`, ... OR `image_1.tif`, `image_2.tif`, ...
- Labels: `Label_1.tif`, `Label_2.tif`, ... OR `label_1.tif`, `label_2.tif`, ...
- Grids (optional per-image): `Grid_1.shp`, `Grid_2.shp`, ...

### Dataset Information
```bash
python Patch_extraction_v2.py info --output /path/to/dataset
```

### Visualization
```bash
python Patch_extraction_v2.py visualize \
    --output /path/to/output_dataset \
    --split train \
    --sample_index 0
```

## Training System

### Unified Training Interface (V2)

The system provides a single entry point (`main_v2.py`) for both segmentation modes with all new features:

```bash
python main_v2.py --mode [binary|multiclass] [OPTIONS]
```

### Basic Training Examples

#### Standard Training with New Features
```bash
# Multi-class with Focal Loss and encoder freezing
python main_v2.py --mode multiclass \
    --classes 5 \
    --dataset_root /path/to/data \
    --model unet++ \
    --loss_type focal_dice \
    --freeze_encoder --freeze_epochs 5 \
    --warmup_epochs 2 \
    --use_amp \
    --log_per_class \
    --class_names background water vegetation buildings roads
```

#### Cross-Validation Training
```bash
# 5-fold cross-validation with per-class metrics
python main_v2.py --mode multiclass \
    --classes 5 \
    --dataset_root /path/to/data \
    --model unet++ \
    --val_strategy kfold \
    --n_splits 5 \
    --loss_type focal \
    --log_per_class
```

## Advanced Training Features (V2)

### Loss Functions

| Loss Type | Description | Best For |
|-----------|-------------|----------|
| `ce` | Cross Entropy only | Balanced datasets |
| `dice` | Dice Loss only | General segmentation |
| `dice_ce` | Dice + Cross Entropy (default) | Balanced approach |
| `focal` | Focal Loss | Class imbalance |
| `focal_dice` | Focal + Dice | Severe imbalance |
| `tversky` | Tversky Loss | Control FP/FN tradeoff |
| `combo` | CE + Dice + Focal | Maximum flexibility |

```bash
# Using Focal Loss with custom parameters
python main_v2.py --loss_type focal --focal_gamma 2.0 --focal_alpha 0.25

# Using Tversky Loss (weight false negatives more)
python main_v2.py --loss_type tversky --tversky_alpha 0.3 --tversky_beta 0.7
```

### Encoder Freezing

Freeze the pretrained encoder to preserve learned features during initial training:

```bash
python main_v2.py --freeze_encoder --freeze_epochs 5
```

This is particularly useful when fine-tuning on small datasets or when the target domain is similar to ImageNet.

### Learning Rate Warmup

Gradually increase learning rate from a small value to the target:

```bash
python main_v2.py --warmup_epochs 3 --warmup_lr 1e-6 --learning_rate 5e-4
```

### Mixed Precision Training

Enable automatic mixed precision for faster training and reduced memory:

```bash
python main_v2.py --use_amp
```

### Per-Class Metrics

Enable detailed per-class IoU logging and visualization:

```bash
python main_v2.py --log_per_class --class_names background water forest urban
```

### Learning Rate Schedulers

```bash
# ReduceLROnPlateau (default)
python main_v2.py --scheduler_type reduce_plateau

# Cosine Annealing
python main_v2.py --scheduler_type cosine

# One-Cycle Policy
python main_v2.py --scheduler_type one_cycle
```

### Available Training Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode` | Segmentation mode: `binary` or `multiclass` | `multiclass` |
| `--dataset_root` | Path to dataset root directory | Required |
| `--model` | Model architecture name | Required |
| `--classes` | Number of classes (for multiclass) | 5 |
| `--val_strategy` | Validation strategy: `split` or `kfold` | `split` |
| `--loss_type` | Loss function type | `dice_ce` |
| `--epochs` | Number of training epochs | 100 |
| `--batch_size` | Batch size | 8 |
| `--learning_rate` | Learning rate | 5e-4 |
| `--encoder_name` | Encoder backbone name | `resnet34` |
| `--pretrained` | Use pretrained encoder weights | True |
| `--freeze_encoder` | Freeze encoder for initial epochs | False |
| `--freeze_epochs` | Number of epochs to keep encoder frozen | 5 |
| `--warmup_epochs` | Number of warmup epochs | 0 |
| `--use_amp` | Enable mixed precision training | False |
| `--log_per_class` | Log per-class IoU metrics | True |
| `--class_names` | Names for each class | Auto-generated |
| `--use_class_weights` | Apply class weights for imbalance | True |
| `--n_splits` | Number of folds for cross-validation | 5 |

### Training Output

During training, the system generates:
```
trained_models/
├── model_best_loss.pth          # Best validation loss checkpoint
├── model_best_iou.pth           # Best validation IoU checkpoint
├── model_final.pth              # Final model with complete config
├── model_training_plot.png      # Training visualization
├── model_training_log.csv       # Complete metrics history (V2)
├── model_per_class_iou.png      # Per-class IoU evolution (V2)
└── model_metrics.json           # Complete metrics in JSON format
```

## Inference on Large Images

The `Predict_large_image_v2.py` script handles prediction on arbitrarily large satellite scenes with automatic encoder detection:

### Basic Prediction
```bash
python Predict_large_image_v2.py --model /path/to/model.pth \
                                 --input /path/to/large_image.tif \
                                 --output /path/to/prediction.tif
```

### Advanced Prediction Options
```bash
# Multi-class with custom parameters
python Predict_large_image_v2.py \
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
| `--encoder_name` | Encoder backbone (auto-detected) | Auto |
| `--patch_size` | Size of prediction patches | 512 |
| `--overlap` | Overlap between patches | 128 |
| `--num_classes` | Number of output classes | Auto-detected |
| `--threshold` | Confidence threshold (binary) | 0.5 |
| `--save_confidence` | Save confidence map | False |
| `--device` | Computation device | `cuda` |

### Seamless Reconstruction

The predictor uses weighted blending to eliminate border artifacts, automatic patch tiling with configurable overlap, geospatial metadata preservation, and nodata value handling from source images.

## Architecture Support

### Available Models

Run the following to see all available models:
```bash
python -c "from model_training_v2 import get_available_models; print(get_available_models())"
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

### Model Checkpoints (V2)

Trained models are saved with comprehensive metadata including all V2 configuration:

```python
{
    'model_state_dict': model_weights,
    'config': {  # V2: Complete TrainingConfig
        'model_name': 'unet++',
        'mode': 'multiclass',
        'in_channels': 10,
        'num_classes': 6,
        'encoder_name': 'resnet34',
        'loss_type': 'focal_dice',
        'freeze_encoder': True,
        'freeze_epochs': 5,
        'warmup_epochs': 2,
        'use_amp': True,
        # ... all other config parameters
    },
    'performance_metrics': {
        'best_val_loss': 0.1234,
        'best_val_iou': 0.7890,
        'per_class_iou': {...}  # V2: Per-class metrics
    }
}
```

### Prediction Outputs

- **Segmentation map**: GeoTIFF with class labels
- **Confidence map** (optional): GeoTIFF with prediction confidence
- **Statistics report**: Console output with class distribution and confidence metrics

## Examples

### Example 1: Land Cover Classification (Multi-class) with V2 Features
```bash
# Extract patches from large scenes (batch mode)
python Patch_extraction_v2.py batch \
    --data_dir ./raw_data \
    --grid grid_polygons.shp \
    --output ./landcover_dataset \
    --patch_size 256 \
    --image_channels 10

# Train with all V2 features
python main_v2.py --mode multiclass \
    --classes 6 \
    --dataset_root ./landcover_dataset \
    --model deeplabv3+ \
    --encoder_name efficientnet-b4 \
    --pretrained \
    --loss_type focal_dice \
    --freeze_encoder --freeze_epochs 5 \
    --warmup_epochs 3 \
    --use_amp \
    --log_per_class \
    --class_names background water forest urban agriculture bare \
    --epochs 200

# Predict on new scene
python Predict_large_image_v2.py \
    --model ./trained_models/model_final.pth \
    --input new_sentinel2_scene.tif \
    --output landcover_prediction.tif \
    --save_confidence
```

### Example 2: Water Body Detection (Binary)
```bash
# Train binary segmentation with Dice loss
python main_v2.py --mode binary \
    --dataset_root ./water_dataset \
    --model unet++ \
    --encoder_name resnet34 \
    --pretrained \
    --loss_type dice \
    --in_channels 10 \
    --epochs 150 \
    --learning_rate 0.0005

# Predict with custom threshold
python Predict_large_image_v2.py \
    --model ./water_models/model_final.pth \
    --input sentinel2_water_scene.tif \
    --output water_mask.tif \
    --threshold 0.3
```

### Example 3: Handling Severe Class Imbalance
```bash
# Use Focal-Dice loss with high gamma and Tversky for FN penalty
python main_v2.py --mode multiclass \
    --classes 5 \
    --dataset_root ./imbalanced_data \
    --model unet++ \
    --loss_type focal_dice \
    --focal_gamma 3.0 \
    --use_class_weights \
    --freeze_encoder --freeze_epochs 10
```

## Best Practices

### Data Preparation
- Normalize images using the 99th percentile method (already implemented)
- Ensure class balance or use `--use_class_weights` for imbalanced datasets
- Use data augmentation (`--data_augmentation`) for small datasets
- Validate spatial alignment between images and masks

### Training Configuration (V2 Recommendations)
- **Start with pretrained encoders** and use `--freeze_encoder` for better transfer learning
- **Use Focal Loss** (`--loss_type focal` or `--loss_type focal_dice`) for imbalanced datasets
- **Enable warmup** (`--warmup_epochs 2-5`) for more stable training
- **Use mixed precision** (`--use_amp`) for faster training on modern GPUs
- Use cross-validation (`--val_strategy kfold`) for reliable performance estimation
- Monitor per-class metrics (`--log_per_class`) to identify underperforming classes

### Inference Settings
- Set appropriate overlap (25-50% of patch size) to avoid border artifacts
- Generate confidence maps (`--save_confidence`) for uncertainty analysis
- Process large images in chunks if memory is limited
- Verify geospatial alignment of output predictions

### Performance Optimization
- Use GPU acceleration for both training and inference
- Adjust patch size based on GPU memory (256-512px recommended)
- Enable mixed precision training (`--use_amp`) for 40-60% speedup
- Use data loaders with pinned memory for faster data transfer

## Troubleshooting

### Common Issues

#### 1. "No images found" error
- **Cause**: Incorrect dataset structure or file extensions
- **Solution**: Verify directory structure and ensure files have `.tif` or `.tiff` extensions

#### 2. CUDA out of memory
- **Cause**: Batch size or patch size too large
- **Solution**: Reduce `--batch_size` or `--patch_size`, enable `--use_amp`

#### 3. Poor prediction quality at patch borders
- **Cause**: Insufficient overlap between patches
- **Solution**: Increase `--overlap` parameter (recommended: 25-50% of patch size)

#### 4. Model fails to load
- **Cause**: Mismatch in model parameters or architecture
- **Solution**: Ensure `--in_channels`, `--num_classes`, and `--encoder_name` match training configuration (V2 auto-detects these from checkpoint)

#### 5. Slow inference speed
- **Cause**: Large patch size or CPU inference
- **Solution**: Reduce patch size, use GPU (`--device cuda`), or enable tiling

#### 6. Class imbalance issues (V2)
- **Cause**: Dominant background class
- **Solution**: Use `--loss_type focal_dice`, increase `--focal_gamma`, enable `--use_class_weights`

### Debug Mode

For detailed debugging, add error tracebacks:
```python
# In model_training_v2.py or Predict_large_image_v2.py
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
