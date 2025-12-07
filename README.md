# SemanticSeg4EO
### Semantic Segmentation for Earth Observation

SemanticSeg4EO is a complete framework for training and applying semantic segmentation models on satellite imagery. It supports both multi-class and binary segmentation tasks and provides a collection of state-of-the-art architectures adapted to Earth Observation data.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Data Structure](#data-structure)
- [Training](#training)
- [Inference](#inference)
- [Supported Architectures](#supported-architectures)
- [Model Format](#model-format)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

---

# Features

## Model Architectures
- **Modified U-Net (“U-Net-ALG”)**:  
  A custom variant of U-Net with *deeper and explicitly hard-coded convolutional blocks*.  
  The model structure can be edited in the source code to adjust architectural depth and feature extraction capacity.  
  The **patch size** and **number of classes** are *pre-initialized* but fully *modifiable via command-line arguments*.

- Standard architectures from *segmentation_models_pytorch (SMP)*:  
  UNet, UNet++, DeepLabV3, DeepLabV3+, FPN, PSPNet, MANet, PAN, LinkNet.

## Segmentation Types
- Multi-class segmentation (up to 6 classes, scalable)
- Binary segmentation (with class imbalance handling)

## Additional Capabilities
- Multi-channel satellite data support (4–10 bands)
- Augmentation pipeline adapted to remote sensing
- Large image tiling and stitching
- Advanced metrics: IoU, F1-score, Recall, Precision
- Automatic checkpointing and early stopping
- TIFF georeferencing preserved on output

---

# Installation

## Requirements
- Python ≥ 3.8  
- PyTorch ≥ 1.10 (CUDA recommended)

## Install
```bash
git clone https://github.com/your-username/SemanticSeg4EO.git
cd SemanticSeg4EO
pip install -r requirements.txt
```

### Minimal `requirements.txt`
```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
rasterio>=1.2.0
tifffile>=2021.7.2
segmentation-models-pytorch>=0.3.0
matplotlib>=3.4.0
tqdm>=4.62.0
scipy>=1.7.0
```

---

# Data Structure

```
dataset_root/
├── Patch/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── validation/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
```

- Images: multi-channel `.tif`  
- Labels: `.tif` masks  
- Multi-class mask values: 0 to N-1  
- Binary mask values: 0 and 1

---

# Training

## Multi-Class Example
```bash
python main.py \
  --dataset_root /path/to/dataset \
  --model unet++ \
  --epochs 100 \
  --batch_size 4 \
  --save_dir ./trained_models \
  --encoder_name resnet34 \
  --pretrained \
  --dropout_rate 0.5 \
  --learning_rate 0.001
```

## Binary Example
```bash
python main_binary.py \
  --dataset_root /path/to/dataset \
  --model unet++ \
  --in_channels 10 \
  --epochs 100 \
  --batch_size 4 \
  --save_dir ./trained_models_binary \
  --encoder_name resnet34 \
  --pretrained \
  --dropout_rate 0.5 \
  --learning_rate 0.001 \
  --data_augmentation \
  --use_class_weights
```

---

# Inference

## Multi-Class (single patch)
```bash
python inference_one_patch.py \
  --model_dir /path/to/models \
  --model_name unet++ \
  --image_path /path/to/image.tif \
  --output_dir ./predictions
```

## Binary (single patch)
```bash
python inference_binary.py \
  --model_dir /path/to/models \
  --model_name unet++ \
  --image_path /path/to/image.tif \
  --threshold 0.3 \
  --output_dir ./predictions
```

## Large Image (multi-class)
```bash
python predict_large_image.py \
  --model_path /path/to/model.pth \
  --input /path/to/large_image.tif \
  --output /path/to/prediction.tif \
  --model_name unet++ \
  --patch_size 512 \
  --overlap 128 \
  --device cuda
```

## Large Image (binary)
```bash
python predict_large_image_binary.py \
  --model /path/to/model.pth \
  --input /path/to/large_image.tif \
  --output /path/to/prediction.tif \
  --patch_size 224 \
  --overlap 64 \
  --threshold 0.5
```

---

# Supported Architectures

## Custom
- Modified U-Net (deeper convolutional backbone, structure editable in code)

## SMP Architectures
UNet, UNet++, DeepLabV3, DeepLabV3+, FPN, PSPNet, PAN, LinkNet, MANet.

## Supported Encoders
ResNet (18–152), EfficientNet, DenseNet, VGG, MobileNet, etc.

---

# Model Format

Generated files include:
- `{model_name}_final_model.pth`
- `{model_name}_best_loss.pth`
- `{model_name}_best_iou.pth`
- `{model_name}_best_combined.pth`
- `{model_name}_metrics.json`
- `{model_name}_training_plot.png`

Metadata includes:
- Architecture and encoder  
- Input channels  
- Number of classes  
- Training configuration  
- Performance metrics  
- Georeferencing information  

---

# Examples

## Binary Water Detection
```bash
python main_binary.py \
  --dataset_root /data/sentinel2_water \
  --model unet++ \
  --in_channels 10 \
  --epochs 150 \
  --batch_size 8 \
  --save_dir ./models/water_detection \
  --encoder_name efficientnet-b3 \
  --pretrained \
  --data_augmentation \
  --use_class_weights \
  --learning_rate 0.0005
```

---

# Best Practices

- Use data augmentation for small datasets.
- Increase patch overlap for large image inference.
- Modify thresholds for binary segmentation depending on sensitivity requirements.
- Reduce batch size or patch size in case of memory limitations.
- Prefer GPU execution for both training and inference.

---

# Troubleshooting

## No images found  
Verify folder structure and `.tif` presence.

## Channel mismatch  
Ensure the value of `--in_channels` matches the image bands.

## Out of memory  
Lower `batch_size` or reduce `patch_size`.

## Patch border artifacts  
Increase `--overlap`.

---

# License
This project is distributed under the MIT License.

---

# Contact
For questions or collaboration requests, please contact:  
**adrien.leguillou@univ-brest.fr**
