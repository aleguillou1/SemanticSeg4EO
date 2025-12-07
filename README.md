# 🌍 SemanticSeg4EO
### Semantic Segmentation for Earth Observation

A complete framework for semantic segmentation of satellite imagery, supporting both multi-class and binary segmentation with state-of-the-art architectures and advanced geospatial features.

---

## 📋 Table of Contents
- [✨ Features](#-features)
- [📥 Installation](#-installation)
- [📁 Data Structure](#-data-structure)
- [🏋️‍♂️ Training](#️-training)
- [🔮 Inference](#-inference)
- [🏗️ Supported Architectures](#️-supported-architectures)
- [💾 Model Format](#-model-format)
- [🚀 Examples](#-examples)
- [💡 Tips and Best Practices](#-tips-and-best-practices)
- [🐛 Troubleshooting](#-troubleshooting)
- [📄 License](#-license)
- [🤝 Contributing](#-contributing)
- [📧 Contact](#-contact)

---

# ✨ Features

## 🏗️ Model Architectures
- **UNet-ALG** (custom architecture with dropout)
- UNet++, DeepLabV3+, FPN, PSPNet, MANet, PAN, LinkNet  
- Support for ResNet, EfficientNet, MobileNet, VGG, DenseNet…
- Fully compatible with **segmentation_models_pytorch (SMP)**

## 🎯 Segmentation Types
- **Multi-Class Segmentation** (up to 6 classes)
- **Binary Segmentation** (with class imbalance handling)

## 🚀 Advanced Features
- Multi-channel satellite data augmentation  
- Multi-metric early stopping  
- Automatic checkpoint saving  
- Georeferenced TIFF support  
- Automatic tiling & stitching for large images  
- Advanced metrics: IoU, F1-score, Precision, Recall  

---

# 📥 Installation

## Requirements
- Python **3.8+**
- PyTorch **1.10+**
- CUDA **11.0+** recommended

## Installation
```bash
git clone https://github.com/your-username/SemanticSeg4EO.git
cd SemanticSeg4EO

pip install -r requirements.txt
```

### Minimum `requirements.txt`
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

# 📁 Data Structure

## Expected Directory Layout
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

### Image Format
- Images: multi-channel `.tif` files (4 or 10 channels)
- Labels: `.tif` segmentation masks  
- Multi-class: values 0 to N-1  
- Binary: 0 or 1  

---

# 🏋️‍♂️ Training

## Multi-Class Segmentation
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

### Main Parameters
- `--model`: unet-alg, unet, unet++, deeplabv3, deeplabv3+, fpn, pspnet, manet, pan, linknet  
- `--encoder_name`: resnet34 by default  
- `--device`: cuda / cpu  

---

## Binary Segmentation
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

### Binary-Specific Parameters
- `--in_channels` (10 for Sentinel-2)
- `--use_class_weights`
- `--patch_size`

---

# 🔮 Inference

## 🔹 Single Patch (Multi-Class)
```bash
python inference_one_patch.py \
  --model_dir /path/to/models \
  --model_name unet++ \
  --image_path /path/to/image.tif \
  --output_dir ./predictions
```

## 🔹 Single Patch (Binary)
```bash
python inference_binary.py \
  --model_dir /path/to/models \
  --model_name unet++ \
  --image_path /path/to/image.tif \
  --threshold 0.3 \
  --output_dir ./predictions
```

---

## 🗺️ Large Image Inference (Multi-Class)
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

## 🗺️ Large Image Inference (Binary)
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

# 🏗️ Supported Architectures

### Custom Architecture
- **unet-alg** (optimized for satellite imagery)

### SMP Architectures
- unet  
- unet++  
- deeplabv3 / deeplabv3+  
- fpn  
- pspnet  
- manet  
- pan  
- linknet  

### Available Encoders
- ResNet 18–152  
- EfficientNet b0–b7  
- DenseNet, VGG, MobileNet  

---

# 💾 Model Format

### Generated Files
- `{model_name}_final_model.pth`
- `{model_name}_best_loss.pth`
- `{model_name}_best_iou.pth`
- `{model_name}_best_combined.pth`
- `{model_name}_metrics.json`
- `{model_name}_training_plot.png`

### Included Metadata
- Architecture  
- Input channels  
- Number of classes  
- Training parameters  
- Performance metrics  
- Georeferencing info  

---

# 🚀 Examples

## 1️⃣ Binary Pipeline (Water Detection)

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

Inference:
```bash
python predict_large_image_binary.py \
  --model ./models/water_detection/unet++_final_model.pth \
  --input /data/region_complete.tif \
  --output ./predictions/water_mask.tif \
  --patch_size 256 \
  --overlap 64 \
  --threshold 0.3
```

---

## 2️⃣ Multi-Class Pipeline  
```bash
python main.py \
  --dataset_root /data/landcover \
  --model deeplabv3+ \
  --epochs 200 \
  --batch_size 6 \
  --save_dir ./models/landcover \
  --encoder_name resnet50 \
  --pretrained \
  --learning_rate 0.001
```

---

# 💡 Tips and Best Practices

### 📉 Small Datasets
- Enable `--data_augmentation`
- Use `--use_class_weights`
- Use pretrained encoders
- Increase dropout rate

### 🗺️ Large Images
- Patch size: 256–512
- Overlap: ≥ 25% of patch size

### ⚖️ Binary Thresholding
- **0.3** → more sensitive  
- **0.5** → balanced default  
- **0.7** → conservative  

### 🚀 Performance Optimization
- Use the GPU (`--device cuda`)
- Reduce `batch_size` if OOM
- Reduce `patch_size` for inference

---

# 🐛 Troubleshooting

### ❌ "No images found"
- Ensure `.tif` format
- Check folder structure

### ❌ "Channel mismatch"
```
Expected input channels: 10, received: 4
```
→ Adjust `--in_channels`

### ❌ Out of Memory (OOM)
- Lower batch size  
- Reduce patch size  

### ❌ Model not found
- Check `.pth` filenames  
- Verify `--model_name`  

### ❌ Patch border artifacts
- Increase `--overlap`

---

# 📝 Example Log Output
```
🚀 STARTING TRAINING
✅ Dataset loaded: 120 training images
✅ Model unet++ built with 4.2M parameters
🎯 Epoch 1/100: Loss: 0.4521, IoU: 0.6789
💾 Checkpoint best_loss saved
...
🎉 TRAINING COMPLETE: Final IoU: 0.8214
```

---

# 📄 License
This project is released under the **MIT License**.

---

# 🤝 Contributing
1. Fork this repository  
2. Create a feature branch  
3. Commit your changes  
4. Push the branch  
5. Open a Pull Request  

---

# 📧 Contact
- Open a GitHub issue  
- Contact the development team  
