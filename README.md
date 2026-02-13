# SemanticSeg4EO – v3

**A Comprehensive Framework for Semantic Segmentation of Earth Observation Imagery**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.10+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

SemanticSeg4EO is a state-of-the-art framework for semantic segmentation of satellite and aerial imagery, supporting both binary and multi-class segmentation through a unified, production-ready codebase. The system integrates modern deep learning architectures specifically optimized for remote sensing applications, with emphasis on methodological transparency, reproducibility, and experimental flexibility.

---

## 🌟 Highlights

- **🎯 Unified Pipeline**: End-to-end workflow from data preparation to inference on large-scale imagery
- **🧠 Modern Architectures**: Support for 15+ state-of-the-art models including SegFormer, ConvNeXt, HRNet, and Swin-UNet
- **🔬 K-Fold Cross-Validation**: Robust validation with confidence intervals and comprehensive statistics
- **🚀 Production-Ready**: Optimized for real-world deployment with large image support and geospatial metadata preservation
- **📊 Advanced Augmentation**: Four levels of data augmentation specifically designed for geospatial data
- **🛰️ Seamless Inference**: Sliding-window prediction with Gaussian blending for artifact-free results

---

## 📋 Table of Contents

- [Overview](#overview)
- [What's New in v3](#whats-new-in-v3)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Patch Extraction](#patch-extraction)
- [Training System](#training-system)
- [Advanced Training Features](#advanced-training-features)
- [Inference on Large Images](#inference-on-large-images)
- [Architecture Support](#architecture-support)
- [Output Format](#output-format)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🔍 Overview

SemanticSeg4EO provides a complete pipeline for Earth Observation (EO) data segmentation:

1. **Data Preparation**: Patch extraction with configurable train/val/test splits from large-scale imagery
2. **Training**: K-Fold cross-validation, multiple modern architectures, sophisticated data augmentation
3. **Inference**: Seamless patch-based prediction on arbitrarily large satellite scenes with georeferencing preservation

The framework is optimized for:
- Multi-spectral satellite imagery (1-20+ channels)
- High-resolution aerial/drone imagery
- Variable patch sizes (224, 512, etc.)
- Both binary and multi-class segmentation tasks
- Imbalanced datasets with class weighting

---

## 🆕 What's New in v3

### 🧠 Modern Architecture Support

- **SegFormer** (B0-B5): Transformer-based encoder with MiT backbone
- **ConvNeXt Family** (Tiny/Small/Base/Large/XLarge): State-of-the-art CNN with UNet decoder
- **UNetFormer**: Hybrid CNN-Transformer decoder for efficient feature fusion
- **HRNet** (W18/W32/W48): High-resolution representation networks
- **Swin-UNet**: Pure transformer U-Net architecture
- **15+ SMP Models**: UNet, UNet++, DeepLabV3+, MANet, FPN, PSPNet, and more

### 🔬 K-Fold Cross-Validation

- Integrated K-Fold validation (`--use_kfold`, `--n_splits`)
- Automatic calculation of mean, standard deviation, and **95% confidence intervals** for all metrics
- Per-fold checkpoints, logs, training curves, and JSON summaries
- Robust performance estimation for small datasets

### 🚀 Advanced Data Augmentation

Four predefined augmentation levels optimized for geospatial data:

| Level | Transformations |
|-------|----------------|
| **None** | No augmentation |
| **Basic** | Horizontal/vertical flip, 90° rotation |
| **Advanced** | + Scale, brightness, contrast, gamma, channel noise |
| **Aggressive** | + Elastic deformation, Gaussian blur, coarse dropout, minority oversampling |
| **Extreme** | + MixUp, CutMix, grid distortion, motion blur, channel shuffle |

### 🛰️ Enhanced Prediction Pipeline

- **Sliding-Window with Overlap**: Handles any-size GeoTIFF with configurable overlap
- **Gaussian Weighting**: Seamless reconstruction without border artifacts
- **Batch GPU Inference**: Optimized memory usage with batch processing
- **Auto-Configuration Detection**: Automatically reads model configuration from checkpoint
- **Confidence Maps**: Optional float32 confidence/probability maps
- **Full Geospatial Preservation**: Maintains CRS, transform, and NoData handling

### ⚙️ Unified Configuration

- Every model accepts configurable `--dropout_rate` (0.0-0.5)
- Checkpoints save complete training configuration
- Predictor automatically detects model architecture and parameters

### 📈 Enhanced Logging & Metrics

- Per-fold CSV logs and training curves
- Confidence intervals and per-class statistics
- Comprehensive JSON output with full experiment metadata

---

## ✨ Key Features

### Core Capabilities

- ✅ **Unified Interface**: Single codebase for binary and multi-class segmentation
- ✅ **K-Fold Cross-Validation**: Robust performance estimation with statistical confidence
- ✅ **Multi-Spectral Support**: Handle 1-20+ channel imagery (RGB, multispectral, hyperspectral)
- ✅ **Class Imbalance Handling**: Automatic class weighting, focal loss, minority oversampling
- ✅ **Production-Ready**: Optimized for real-world deployment scenarios

### Training Features

- 🎯 **15+ Architectures**: From classic UNet to modern transformers
- 🎯 **10+ Loss Functions**: Cross-entropy, Dice, Focal, Tversky, and combinations
- 🎯 **Smart Optimization**: Learning rate warmup, encoder freezing, mixed precision (AMP)
- 🎯 **Flexible Augmentation**: Four levels from basic to extreme transformations
- 🎯 **Advanced Scheduling**: Cosine annealing, ReduceLROnPlateau, OneCycleLR

### Inference Features

- 🔮 **Large Image Support**: Process images of any size with sliding-window approach
- 🔮 **Gaussian Blending**: Weighted fusion eliminates tiling artifacts
- 🔮 **Batch Processing**: GPU-optimized batch inference
- 🔮 **Geospatial Preservation**: Full CRS, transform, and metadata preservation
- 🔮 **Confidence Maps**: Export prediction confidence for uncertainty quantification

---

## 🔧 Installation

### System Requirements

- **Python**: ≥ 3.8
- **PyTorch**: ≥ 1.10 (CUDA recommended for GPU acceleration)
- **GPU**: Recommended for training and large-scale inference
- **RAM**: 16GB+ recommended for large imagery

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/aleguillou1/SemanticSeg4EO.git
cd SemanticSeg4EO

# Install dependencies
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)

```txt
# Core Deep Learning
torch>=1.10.0
torchvision>=0.11.0

# Segmentation Models
segmentation-models-pytorch>=0.3.0
transformers>=4.21.0          # For SegFormer
timm>=0.6.0                    # For ConvNeXt, HRNet, Swin, UNetFormer

# Geospatial Processing
rasterio>=1.3.0
geopandas>=0.12.0
tifffile>=2022.5.4

# Computer Vision & Utils
opencv-python>=4.5.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
tqdm>=4.64.0
```

### Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# List available models
python Model_training.py --list-models

# List available loss functions
python Model_training.py --list-losses
```

---

## 🚀 Quick Start

### 1. Prepare Your Data

Organize your satellite imagery and labels:

```
raw_data/
├── Image_1.tif      # Multi-band satellite image
├── Label_1.tif      # Single-band label mask
├── Image_2.tif
├── Label_2.tif
└── grid.shp         # Vector grid for patch extraction
```

### 2. Extract Patches

```bash
# Single image mode
python Patch_extraction.py single \
    --image ortho.tif --label mask.tif --grid grid.shp \
    --output ./dataset --patch_size 224 --image_channels 4

# Batch mode (multiple images)
python Patch_extraction.py batch \
    --data_dir ./raw_data --grid grid.shp --output ./dataset \
    --patch_size 224 --image_channels 4 --recursive
```

### 3. Train a Model

```bash
# Multi-class segmentation with UNet++ and ResNet34
python Model_training.py \
    --mode multiclass --classes 5 --dataset_root ./dataset \
    --model unet++ --encoder resnet34 --aug_level advanced \
    --loss_type focal_dice --use_class_weights --epochs 100

# Binary segmentation with ConvNeXt-Tiny
python Model_training.py \
    --mode binary --dataset_root ./dataset \
    --model unet --encoder convnext_tiny --aug_level aggressive \
    --loss_type binary_focal_dice --epochs 150

# K-Fold cross-validation
python Model_training.py \
    --mode multiclass --classes 5 --dataset_root ./dataset \
    --model segformer-b2 --use_kfold --n_splits 5 \
    --aug_level advanced --loss_type focal_dice
```

### 4. Predict on Large Images

```bash
python Predict_large_image.py \
    --model_path trained_models/model_best_iou.pth \
    --input large_satellite_image.tif --output prediction.tif \
    --model_name unet++ --encoder resnet34 \
    --in_channels 4 --num_classes 5 \
    --patch_size 512 --overlap 128 --save_confidence
```

---

## 📊 Dataset Preparation

### Required Format

#### Images
- **Format**: Multi-band GeoTIFF
- **Channels**: 1-20+ bands (e.g., RGB, multispectral, hyperspectral)
- **Examples**: 
  - 3-band: RGB aerial imagery
  - 4-band: RGB + NIR (Sentinel-2, drone)
  - 10-band: Full Sentinel-2 spectral bands

#### Labels
- **Format**: Single-band GeoTIFF
- **Binary Mode**: 
  - `0` = Background
  - `1` = Foreground
- **Multi-Class Mode**: 
  - `0 to N-1` = Class indices (e.g., 0=background, 1=water, 2=vegetation, 3=urban)

#### Spatial Alignment
- Images and labels **must** be spatially aligned
- Same CRS (Coordinate Reference System)
- Same spatial extent and resolution

### Expected Directory Structure

After patch extraction, your dataset should look like:

```
dataset_root/
└── Patch/
    ├── train/
    │   ├── images/     # Training image patches
    │   │   ├── patch_001.tif
    │   │   ├── patch_002.tif
    │   │   └── ...
    │   └── labels/     # Corresponding label patches
    │       ├── patch_001.tif
    │       ├── patch_002.tif
    │       └── ...
    ├── validation/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

---

## ✂️ Patch Extraction

The `Patch_extraction.py` script cuts large satellite scenes into training patches using a vector grid.

### Single Image Mode

Extract patches from a single image-label pair:

```bash
python Patch_extraction.py single \
    --image satellite_image.tif \
    --label ground_truth.tif \
    --grid patch_grid.shp \
    --output ./dataset \
    --patch_size 224 \
    --image_channels 10 \
    --train_ratio 0.75 \
    --val_ratio 0.2 \
    --test_ratio 0.05 \
    --random_seed 42
```

### Batch Mode

Process multiple image-label pairs at once:

```bash
python Patch_extraction.py batch \
    --data_dir ./raw_satellite_data \
    --grid shared_grid.shp \
    --output ./dataset \
    --patch_size 256 \
    --image_channels 4 \
    --recursive \
    --verbose
```

**Naming Convention for Batch Mode:**
- Images: `Image_1.tif`, `Image_2.tif`, ... (or `image_1.tif`, case-insensitive)
- Labels: `Label_1.tif`, `Label_2.tif`, ... (or `label_1.tif`, case-insensitive)
- Grids (optional): `Grid_1.shp`, `Grid_2.shp`, ... (with `--use_per_image_grid`)

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--patch_size` | Size of square patches in pixels | 224 |
| `--image_channels` | Number of spectral bands | 4 |
| `--train_ratio` | Proportion for training | 0.75 |
| `--val_ratio` | Proportion for validation | 0.20 |
| `--test_ratio` | Proportion for testing | 0.05 |
| `--random_seed` | For reproducible splits | None |
| `--interpolation` | Resampling method | bilinear |
| `--no_compression` | Disable LZW compression | False |

### Utility Commands

```bash
# Display dataset information
python Patch_extraction.py info --output ./dataset

# Visualize a sample patch
python Patch_extraction.py visualize --output ./dataset --split train --sample_index 0
```

All patch information is saved in `patch_metadata.json` within the output directory.

---

## 🎓 Training System

The unified entry point is `Model_training.py`, which accepts extensive configuration options.

### Basic Training

```bash
# Standard multi-class training
python Model_training.py \
    --mode multiclass --classes 5 \
    --dataset_root ./dataset \
    --model unet++ \
    --encoder resnet34 \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4
```

### K-Fold Cross-Validation

For robust performance estimation, especially with limited data:

```bash
python Model_training.py \
    --mode multiclass --classes 5 \
    --dataset_root ./dataset \
    --model segformer-b2 \
    --use_kfold --n_splits 5 \
    --aug_level aggressive \
    --loss_type focal_dice \
    --batch_size 4
```

**K-Fold Output:**
- Individual fold checkpoints, logs, and metrics
- Aggregated statistics with mean ± std and 95% CI
- Cross-validation results in `cv_results_*.json`

### Using Modern Architectures

```bash
# SegFormer-B3 (Transformer)
python Model_training.py \
    --mode multiclass --classes 6 \
    --dataset_root ./dataset \
    --model segformer-b3 \
    --aug_level aggressive \
    --batch_size 4

# ConvNeXt-Small (requires --model unet)
python Model_training.py \
    --mode binary \
    --dataset_root ./dataset \
    --model unet --encoder convnext_small \
    --aug_level advanced \
    --loss_type binary_focal_dice

# HRNet-W32
python Model_training.py \
    --mode multiclass --classes 4 \
    --dataset_root ./dataset \
    --model hrnet-w32 \
    --aug_level aggressive
```

---

## 🔥 Advanced Training Features

### Augmentation Levels

Set with `--aug_level [none|basic|advanced|aggressive|extreme]`:

#### **None**
- No augmentation applied
- Use for testing or when data is already diverse

#### **Basic**
- Horizontal/vertical flips (0.5 probability each)
- 90° rotations (0.5 probability)
- Ideal for aerial imagery (no preferred orientation)

#### **Advanced** (Recommended)
- All basic transformations
- Scale variations (0.85-1.15x)
- Brightness adjustments (±15%)
- Contrast variations (±15%)
- Gamma correction (0.85-1.15)
- Channel noise (simulates sensor noise)

#### **Aggressive**
- All advanced transformations
- Elastic deformations
- Gaussian blur
- Coarse dropout
- Minority class oversampling
- Suitable for limited datasets

#### **Extreme**
- All aggressive transformations
- MixUp and CutMix
- Grid distortion
- Motion blur
- Channel shuffle
- Maximum augmentation for very small datasets

### Loss Functions

The framework automatically adapts loss functions based on mode (binary/multiclass):

| Loss Type | Binary Version | Multi-Class Version | Use Case |
|-----------|---------------|---------------------|----------|
| `ce` | `bce` | `ce` | Standard cross-entropy |
| `dice` | `binary_dice` | `dice` | Overlap-based metric |
| `dice_ce` | `binary_dice_bce` | `dice_ce` | Combined approach |
| `focal` | `binary_focal` | `focal` | Class imbalance |
| `focal_dice` | `binary_focal_dice` | `focal_dice` | **Recommended for imbalance** |
| `tversky` | `binary_tversky` | `tversky` | FP/FN balance control |
| `focal_tversky` | `binary_focal_tversky` | `focal_tversky` | Advanced imbalance |
| `combo` | N/A | `combo` | CE + Dice + Focal |

**Example:**

```bash
# Focal + Dice for imbalanced datasets
python Model_training.py \
    --mode multiclass --classes 5 \
    --dataset_root ./dataset \
    --model unet++ --encoder efficientnet-b4 \
    --loss_type focal_dice --focal_gamma 2.5 --focal_alpha 0.25 \
    --use_class_weights
```

### Optimization Strategies

#### Encoder Freezing

Freeze the encoder for initial epochs to stabilize training:

```bash
--freeze_encoder --freeze_epochs 5
```

#### Learning Rate Warmup

Gradually increase learning rate at the start:

```bash
--warmup_epochs 3 --warmup_lr 1e-6
```

#### Learning Rate Scheduling

```bash
# Cosine annealing (default)
--scheduler cosine

# Reduce on plateau
--scheduler reduce_plateau --patience 10

# One-cycle policy
--scheduler one_cycle
```

#### Mixed Precision Training

Enable automatic mixed precision for faster training:

```bash
--use_amp
```

### Class Weighting

Automatically weight classes based on frequency:

```bash
--use_class_weights
```

### Per-Class Logging

Track metrics for each class individually:

```bash
--log_per_class --class_names background water vegetation urban
```

### Complete Advanced Example

```bash
python Model_training.py \
    --mode multiclass --classes 6 \
    --dataset_root ./dataset \
    --model unet++ --encoder efficientnet-b4 \
    --aug_level aggressive \
    --loss_type focal_dice --focal_gamma 2.5 \
    --use_class_weights \
    --freeze_encoder --freeze_epochs 5 \
    --warmup_epochs 3 --warmup_lr 1e-6 \
    --scheduler cosine \
    --use_amp \
    --batch_size 8 --epochs 150 \
    --dropout_rate 0.3 \
    --log_per_class --class_names bg water veg urban bare cloud
```

---

## 🔮 Inference on Large Images

The `Predict_large_image.py` script handles images of any size while preserving geospatial metadata.

### Basic Usage

```bash
python Predict_large_image.py \
    --model_path trained_models/best_model.pth \
    --input large_satellite_image.tif \
    --output prediction_mask.tif \
    --model_name unet++ \
    --encoder resnet34 \
    --in_channels 4 \
    --num_classes 5
```

### Advanced Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--patch_size` | Size of prediction patches | 224 |
| `--overlap` | Overlap between patches (pixels) | 112 |
| `--batch_size` | GPU batch size for inference | 4 |
| `--threshold` | Binary segmentation threshold | 0.5 |
| `--save_confidence` | Save confidence/probability map | False |
| `--device` | Inference device | cuda |
| `--output_nodata` | NoData value for output | 255 |

### Sliding-Window Approach

The predictor uses an intelligent sliding-window strategy:

1. **Patch Grid Generation**: Creates overlapping patches across the entire image
2. **Batch Processing**: Processes multiple patches simultaneously on GPU
3. **Gaussian Weighting**: Weights center pixels higher than edges
4. **Seamless Fusion**: Blends overlapping predictions to eliminate artifacts

**Recommended Overlap:**
- Use 25-50% of `--patch_size` for best results
- Larger overlap = smoother transitions but slower inference
- Example: `--patch_size 512 --overlap 128` (25% overlap)

### Examples

```bash
# Binary water detection
python Predict_large_image.py \
    --model_path water_model.pth \
    --input sentinel2_scene.tif \
    --output water_mask.tif \
    --model_name unet --encoder convnext_small \
    --in_channels 10 --num_classes 1 \
    --patch_size 256 --overlap 64 \
    --threshold 0.3 --save_confidence

# Multi-class land cover
python Predict_large_image.py \
    --model_path landcover_model.pth \
    --input aerial_ortho.tif \
    --output landcover.tif \
    --model_name segformer-b3 \
    --in_channels 4 --num_classes 7 \
    --patch_size 512 --overlap 128 \
    --batch_size 8 --save_confidence

# Large batch for high-memory GPU
python Predict_large_image.py \
    --model_path model.pth \
    --input huge_image.tif \
    --output result.tif \
    --model_name manet --encoder efficientnet-b3 \
    --in_channels 3 --num_classes 4 \
    --patch_size 224 --overlap 112 \
    --batch_size 16
```

### Automatic Configuration Detection

The predictor can automatically read model configuration from checkpoints saved by the training script:

```bash
# Minimal command - auto-detects architecture, channels, classes
python Predict_large_image.py \
    --model_path trained_models/model_best_iou.pth \
    --input image.tif \
    --output prediction.tif
```

---

## 🏗️ Architecture Support

### Full Model List

#### Segmentation Models PyTorch (SMP)

Compatible with all standard encoders (ResNet, EfficientNet, etc.):

- **UNet**: Classic encoder-decoder with skip connections
- **UNet++**: Nested U-Net with dense skip pathways
- **DeepLabV3+**: Atrous spatial pyramid pooling
- **DeepLabV3**: ASPP without decoder
- **MANet**: Multi-scale attention network
- **FPN**: Feature pyramid network
- **PAN**: Pyramid attention network
- **PSPNet**: Pyramid scene parsing network
- **LinkNet**: Efficient encoder-decoder

#### Transformer-Based Models

- **SegFormer-B0**: Lightweight transformer (3.8M params)
- **SegFormer-B1**: Balanced model (13.7M params)
- **SegFormer-B2**: Mid-size model (27.4M params)
- **SegFormer-B3**: Large model (47.3M params)
- **SegFormer-B4**: Very large model (64.1M params)
- **SegFormer-B5**: Largest variant (84.7M params)

#### Hybrid & Modern Architectures

- **UNetFormer**: CNN-Transformer hybrid decoder
- **HRNet-W18**: High-resolution network (21M params)
- **HRNet-W32**: Mid-size HRNet (41M params)
- **HRNet-W48**: Large HRNet (77M params)
- **Swin-UNet**: Pure transformer U-Net

#### ConvNeXt Family (UNet Only)

⚠️ **Important**: ConvNeXt encoders work only with `--model unet`

- **ConvNeXt-Tiny**: Modern CNN, efficient (28M params) ⭐
- **ConvNeXt-Small**: Balanced performance (50M params) ⭐⭐
- **ConvNeXt-Base**: High performance (89M params) ⭐⭐⭐
- **ConvNeXt-Large**: State-of-the-art (198M params)
- **ConvNeXt-XLarge**: Maximum capacity (350M params)

### Supported Encoders

For SMP models and UNetFormer, you can use:

#### ResNet Family
- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`

#### EfficientNet Family
- `efficientnet-b0` through `efficientnet-b7`
- Recommended: `efficientnet-b3` or `efficientnet-b4`

#### SENet & ResNeXt
- `se_resnext50_32x4d`, `senet154`

#### MobileNet
- `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`

#### DenseNet
- `densenet121`, `densenet169`, `densenet201`, `densenet264`

#### VGG
- `vgg11`, `vgg13`, `vgg16`, `vgg19`

And many more from the `timm` library!

### Model Selection Guide

| Task | Recommended Model | Encoder | Rationale |
|------|------------------|---------|-----------|
| **Binary (small dataset)** | UNet | `convnext_tiny` | Efficient, modern architecture |
| **Binary (large dataset)** | UNet++ | `efficientnet-b4` | Strong performance, balanced |
| **Multi-class (3-5 classes)** | SegFormer-B2 | N/A | Excellent for moderate classes |
| **Multi-class (6+ classes)** | SegFormer-B3/B4 | N/A | Higher capacity for complexity |
| **High-resolution** | HRNet-W32 | N/A | Maintains resolution throughout |
| **Limited GPU memory** | UNet | `resnet34` | Memory-efficient |
| **Maximum performance** | UNet++ | `convnext_base` | State-of-the-art results |

---

## 📁 Output Format

### Model Checkpoints

All checkpoints saved by the training script contain:

```python
{
    'model_state_dict': ...,           # Model weights
    'config': {...},                    # Full TrainingConfig as dict
    'metrics': {...},                   # Best validation scores
    'optimizer_state_dict': ...,        # For resuming training
    'scheduler_state_dict': ...,        # For resuming training
    'epoch': int,                       # Epoch number
    'augmentation_config': {...}        # Augmentation settings
}
```

**Saved Checkpoints:**
- `*_best_loss.pth`: Best validation loss
- `*_best_iou.pth`: Best mean IoU (recommended for inference)
- `*_best_combined.pth`: Best combined metric
- `*_latest.pth`: Most recent epoch

### Training Logs

For each training run:

```
trained_models/
└── modelname_YYYYMMDD_HHMMSS/
    ├── model_best_iou.pth
    ├── model_best_loss.pth
    ├── model_latest.pth
    ├── training_log.csv              # Epoch-by-epoch metrics
    ├── training_curves.png           # Loss and IoU plots
    └── metrics.json                  # Full experiment summary
```

**CSV Log Format:**
```csv
epoch,train_loss,train_miou,val_loss,val_miou,val_f1,lr,time
1,0.4523,0.6234,0.3891,0.6789,0.7234,0.0001,45.2
2,0.3821,0.7012,0.3456,0.7234,0.7656,0.0001,44.8
...
```

**JSON Metrics:**
```json
{
    "config": {...},
    "training_history": [...],
    "best_epoch": 67,
    "best_metrics": {
        "val_loss": 0.2134,
        "val_miou": 0.8567,
        "val_f1": 0.8834
    },
    "test_metrics": {
        "test_loss": 0.2201,
        "test_miou": 0.8512,
        "per_class_iou": [0.92, 0.87, 0.81, 0.79, 0.86]
    }
}
```

### K-Fold Output Structure

```
trained_models/kfold_YYYYMMDD_HHMMSS/
├── fold_0/
│   ├── model_best_iou.pth
│   ├── training_log.csv
│   ├── training_curves.png
│   └── fold_0_metrics.json
├── fold_1/
│   └── ...
├── fold_2/
│   └── ...
├── fold_3/
│   └── ...
├── fold_4/
│   └── ...
└── cv_results.json                # Aggregated statistics
```

**Cross-Validation Results:**
```json
{
    "cv_stats": {
        "mean_iou": 0.8512,
        "std_iou": 0.0234,
        "ci_95_iou": [0.8278, 0.8746],    # 95% confidence interval
        "mean_f1": 0.8834,
        "std_f1": 0.0189,
        "ci_95_f1": [0.8645, 0.9023],
        "fold_results": [
            {"fold": 0, "iou": 0.8456, "f1": 0.8723},
            {"fold": 1, "iou": 0.8534, "f1": 0.8889},
            ...
        ]
    }
}
```

### Prediction Outputs

```
predictions/
├── prediction.tif                 # Main segmentation mask (uint8)
└── prediction_confidence.tif      # Optional confidence map (float32)
```

**Main Mask:**
- **Format**: GeoTIFF
- **Type**: uint8
- **Values**: 
  - Binary: 0 (background), 1 (foreground)
  - Multi-class: 0 to N-1 (class indices)
- **NoData**: 255 (default, configurable)
- **Compression**: LZW
- **Geospatial**: Full CRS and transform preservation

**Confidence Map:**
- **Format**: GeoTIFF
- **Type**: float32
- **Values**: 
  - Binary: Probability of foreground class
  - Multi-class: Maximum class probability
- **Range**: 0.0 to 1.0
- **NoData**: -9999.0

---

## 💡 Examples

### Example 1: Land Cover Classification

**Scenario**: 6-class land cover mapping from Sentinel-2 imagery

```bash
# Step 1: Extract patches (10-band Sentinel-2)
python Patch_extraction.py batch \
    --data_dir ./sentinel2_scenes \
    --grid patch_grid.shp \
    --output ./landcover_dataset \
    --patch_size 256 \
    --image_channels 10 \
    --recursive

# Step 2: Train with K-Fold validation
python Model_training.py \
    --mode multiclass --classes 6 \
    --dataset_root ./landcover_dataset \
    --model unet++ --encoder efficientnet-b4 \
    --aug_level aggressive \
    --loss_type focal_dice --focal_gamma 2.5 \
    --use_class_weights \
    --use_kfold --n_splits 5 \
    --freeze_encoder --freeze_epochs 5 \
    --batch_size 8 --epochs 150 \
    --class_names background water vegetation urban bare cloud

# Step 3: Predict on new scene
python Predict_large_image.py \
    --model_path trained_models/kfold_*/fold_0/model_best_iou.pth \
    --input new_sentinel2_scene.tif \
    --output landcover_map.tif \
    --model_name unet++ --encoder efficientnet-b4 \
    --in_channels 10 --num_classes 6 \
    --patch_size 512 --overlap 128 \
    --save_confidence
```

### Example 2: Binary Water Detection

**Scenario**: High-precision water body detection from 4-band drone imagery

```bash
# Step 1: Extract patches
python Patch_extraction.py single \
    --image drone_ortho.tif \
    --label water_mask.tif \
    --grid grid.shp \
    --output ./water_dataset \
    --patch_size 224 \
    --image_channels 4

# Step 2: Train with modern architecture
python Model_training.py \
    --mode binary \
    --dataset_root ./water_dataset \
    --model unet --encoder convnext_small \
    --aug_level advanced \
    --loss_type binary_focal_dice --focal_gamma 3.0 \
    --use_class_weights \
    --batch_size 16 --epochs 200 \
    --warmup_epochs 3

# Step 3: Predict with custom threshold
python Predict_large_image.py \
    --model_path trained_models/unet_best_iou.pth \
    --input new_drone_image.tif \
    --output water_prediction.tif \
    --model_name unet --encoder convnext_small \
    --in_channels 4 --num_classes 1 \
    --threshold 0.35 --save_confidence
```

### Example 3: Severe Class Imbalance

**Scenario**: Rare object detection (e.g., solar panels, <2% of pixels)

```bash
python Model_training.py \
    --mode binary \
    --dataset_root ./solar_panels \
    --model segformer-b3 \
    --aug_level extreme \
    --loss_type binary_focal_dice --focal_gamma 4.0 --focal_alpha 0.75 \
    --use_class_weights \
    --freeze_encoder --freeze_epochs 10 \
    --warmup_epochs 5 \
    --batch_size 4 --epochs 300 \
    --patience 50
```

### Example 4: High-Resolution Aerial Imagery

**Scenario**: Fine-grained segmentation from 0.1m resolution aerial imagery

```bash
# Step 1: Extract larger patches
python Patch_extraction.py batch \
    --data_dir ./aerial_0.1m \
    --grid grid.shp \
    --output ./aerial_dataset \
    --patch_size 512 \
    --image_channels 3

# Step 2: Train HRNet for high-resolution
python Model_training.py \
    --mode multiclass --classes 8 \
    --dataset_root ./aerial_dataset \
    --model hrnet-w32 \
    --aug_level aggressive \
    --loss_type combo \
    --use_class_weights \
    --batch_size 4 --epochs 150 \
    --patch_size 512

# Step 3: Predict with large overlap for smooth results
python Predict_large_image.py \
    --model_path trained_models/hrnet-w32_best_iou.pth \
    --input aerial_scene.tif \
    --output segmentation.tif \
    --model_name hrnet-w32 \
    --in_channels 3 --num_classes 8 \
    --patch_size 512 --overlap 256 \
    --batch_size 2
```

---

## 🎯 Best Practices

### Data Preparation

1. **Normalization**: Images are automatically normalized using percentile-99 method (handles outliers better than min-max)
2. **Patch Size**: 
   - 224 or 256: Standard for most tasks
   - 512: High-resolution imagery or fine details
   - Power of 2 recommended for most architectures
3. **Splits**: Default 75/20/5 (train/val/test) works well for most cases
4. **Random Seed**: Always set for reproducible experiments

### Handling Class Imbalance

1. **Always use** `--use_class_weights` for imbalanced datasets
2. **Recommended loss**: `focal_dice` or `focal_tversky`
3. **Focal gamma**: 
   - 2.0: Moderate imbalance (10:1)
   - 2.5-3.0: Severe imbalance (50:1)
   - 4.0+: Extreme imbalance (100:1+)
4. **Augmentation**: Use `aggressive` or `extreme` for minority classes
5. **Minority oversampling**: Included in `aggressive` and `extreme` levels

### Small Datasets

1. **K-Fold CV**: Use `--use_kfold` for robust evaluation
2. **Augmentation**: Start with `aggressive`, try `extreme` if needed
3. **Encoder freezing**: `--freeze_encoder --freeze_epochs 5-10`
4. **Pretrained weights**: Always enabled by default
5. **Regularization**: Increase `--dropout_rate` to 0.4-0.5

### Large Images (Prediction)

1. **Overlap**: Set to 25-50% of patch size
   - Smaller overlap: Faster but possible artifacts
   - Larger overlap: Slower but smoother results
2. **Batch size**: Maximize based on GPU memory
3. **Patch size**: Match training patch size or use multiples
4. **Confidence maps**: Enable to identify uncertain regions

### GPU Memory Management

1. **Reduce batch size**: Start with 8, reduce to 4 or 2 if needed
2. **Reduce patch size**: Try 224 instead of 512
3. **Enable AMP**: `--use_amp` for mixed precision (2x memory savings)
4. **Gradient accumulation**: Not currently supported (future feature)

### Model Selection

1. **Binary tasks**: UNet with `convnext_tiny` or `resnet34`
2. **Multi-class (3-5 classes)**: SegFormer-B2 or UNet++
3. **Multi-class (6+ classes)**: SegFormer-B3/B4 or UNet++ with EfficientNet-B4
4. **Limited GPU**: UNet with `resnet34`
5. **Maximum accuracy**: UNet++ with `convnext_base` or `efficientnet-b7`

### Cross-Validation

1. **Use K-Fold** for datasets with <1000 patches
2. **Number of folds**: 5 is standard, 10 for very small datasets
3. **Evaluation metric**: Use `best_iou` checkpoint for final inference
4. **Confidence intervals**: Report mean ± std and 95% CI

---

## 🔧 Troubleshooting

### Common Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| **No images found** | Wrong dataset path or file extensions | Verify directory structure and `.tif`/`.tiff` extensions |
| **CUDA out of memory** | Batch/patch size too large | Reduce `--batch_size` or `--patch_size`, enable `--use_amp` |
| **Poor prediction at borders** | Insufficient overlap | Increase `--overlap` to 50% of patch size |
| **Model fails to load** | Architecture mismatch | Ensure `--in_channels`, `--num_classes` match training |
| **Slow inference** | CPU mode or large patches | Use `--device cuda`, reduce `--patch_size` |
| **Class imbalance issues** | Dominant background class | Use `--loss_type focal_dice` with `--focal_gamma 2-3` and `--use_class_weights` |
| **K-Fold error** | Insufficient samples | Increase `--n_splits` or use more subdirectories |
| **Training loss NaN** | Learning rate too high | Reduce `--lr` to 1e-5 or enable `--warmup_epochs` |
| **Validation loss increases** | Overfitting | Increase augmentation level, add `--dropout_rate 0.4` |
| **ConvNeXt error** | Wrong model combination | ConvNeXt requires `--model unet`, not unet++ or others |

### Debug Mode

Enable full error traces:

```python
import traceback
try:
    # Your code
except Exception as e:
    traceback.print_exc()
```

### Performance Optimization

**Training Speed:**
1. Enable mixed precision: `--use_amp`
2. Increase batch size (within GPU limits)
3. Use fewer workers: `--num_workers 2-4`
4. Reduce validation frequency (modify code)

**Inference Speed:**
1. Use batch inference: `--batch_size 8-16`
2. Reduce overlap: `--overlap 64` for 224px patches
3. Use GPU: `--device cuda`
4. Smaller models: UNet with `resnet34` instead of larger variants

### Validation

**Sanity Checks:**

```bash
# Verify dataset structure
python Patch_extraction.py info --output ./dataset

# Check model list
python Model_training.py --list-models

# Test single epoch
python Model_training.py ... --epochs 1

# Visualize augmentations (add custom code to visualize)
python Patch_extraction.py visualize --output ./dataset --split train
```

---

## 📚 Citation

If you use SemanticSeg4EO in your research, please cite:

```bibtex
@software{semanticseg4eo2024,
  author = {Le Guillou, Adrien},
  title = {SemanticSeg4EO: A Unified Framework for Semantic Segmentation of Earth Observation Imagery},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/aleguillou1/SemanticSeg4EO},
  version = {3.0}
}
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Adrien Leguillou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 📧 Contact

**Adrien Leguillou**  
Research Engineer – LETG (Littoral, Environnement, Télédétection, Géomatique)  
University of Brest, France

- **Email**: adrien.leguillou@univ-brest.fr
- **GitHub**: [@aleguillou1](https://github.com/aleguillou1)
- **Repository**: [SemanticSeg4EO](https://github.com/aleguillou1/SemanticSeg4EO)

For questions, collaborations, bug reports, or feature requests:
1. Open an issue on [GitHub Issues](https://github.com/aleguillou1/SemanticSeg4EO/issues)
2. Contact via email for technical support or research collaborations

---

## 🙏 Acknowledgments

This framework builds upon several outstanding open-source projects:

- **[Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)** - Pavel Yakubovskiy (SMP library)
- **[PyTorch](https://pytorch.org/)** - Facebook AI Research
- **[Transformers](https://github.com/huggingface/transformers)** - Hugging Face (SegFormer implementation)
- **[timm](https://github.com/rwightman/pytorch-image-models)** - Ross Wightman (PyTorch Image Models)
- **[Rasterio](https://rasterio.readthedocs.io/)** - Mapbox (Geospatial I/O)
- **[GDAL](https://gdal.org/)** - OSGeo (Geospatial Data Abstraction Library)
- **[GeoPandas](https://geopandas.org/)** - GeoPandas developers

Special thanks to:
- The remote sensing and Earth observation community for datasets and methodologies
- LETG laboratory for research support and infrastructure
- All contributors and users who have provided feedback and improvements

---

## 🌍 Applications

SemanticSeg4EO has been successfully applied to:

- **Land Cover Mapping**: Multi-class classification of satellite imagery
- **Water Body Detection**: Binary segmentation for hydrology studies
- **Urban Monitoring**: Building and infrastructure extraction
- **Agricultural Analysis**: Crop type mapping and field boundary detection
- **Forest Monitoring**: Tree cover and deforestation tracking
- **Coastal Studies**: Shoreline extraction and coastal change detection
- **Disaster Response**: Flood mapping and damage assessment

---

## 🔮 Future Developments

Planned features for upcoming versions:

- [ ] Support for additional architectures (Mask2Former, SegNext)
- [ ] Multi-GPU training with DataParallel/DistributedDataParallel
- [ ] Uncertainty quantification methods
- [ ] Active learning workflows
- [ ] Integration with cloud platforms (AWS, GCP, Azure)
- [ ] Streamlit/Gradio web interface
- [ ] Pre-trained weights on common remote sensing datasets
- [ ] Support for temporal/multi-date imagery
- [ ] Change detection capabilities

---

## ⚡ Quick Reference

### Training Command Template

```bash
python Model_training.py \
    --mode [binary|multiclass] \
    --classes N \
    --dataset_root ./path/to/dataset \
    --model [unet|unet++|segformer-b2|...] \
    --encoder [resnet34|efficientnet-b3|convnext_tiny|...] \
    --aug_level [none|basic|advanced|aggressive|extreme] \
    --loss_type [focal_dice|dice_ce|focal|...] \
    --use_class_weights \
    --batch_size 8 \
    --epochs 100
```

### Prediction Command Template

```bash
python Predict_large_image.py \
    --model_path model.pth \
    --input image.tif \
    --output prediction.tif \
    --model_name [architecture] \
    --encoder [backbone] \
    --in_channels N \
    --num_classes N \
    --patch_size 224 \
    --overlap 112
```

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

[Report Bug](https://github.com/aleguillou1/SemanticSeg4EO/issues) · 
[Request Feature](https://github.com/aleguillou1/SemanticSeg4EO/issues) · 
[Documentation](https://github.com/aleguillou1/SemanticSeg4EO/wiki)

</div>
