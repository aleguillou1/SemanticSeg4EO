# 🌍 SemanticSeg4EO
### Semantic Segmentation for Earth Observation

Un framework complet pour la segmentation sémantique d’images satellites, compatible multiclasse et binaire, avec des architectures state-of-the-art.

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
- **UNet-ALG** (architecture custom avec dropout)
- UNet++, DeepLabV3+, FPN, PSPNet, MANet, PAN, LinkNet  
- Encoders : ResNet, EfficientNet, MobileNet, VGG, DenseNet…
- Compatible **segmentation_models_pytorch**

## 🎯 Segmentation Types
- **Multi-Class Segmentation** (jusqu’à 6 classes)
- **Binary Segmentation** (gestion du déséquilibre)

## 🚀 Advanced Features
- Augmentation multi-canaux optimisée satellite
- Early stopping multi-métriques
- Checkpoints automatiques
- Support géoréférencement (.tif)
- Tiling / reconstruction automatique grandes images
- Metrics avancées : IoU, F1, Precision, Recall

---

# 📥 Installation

## Prérequis
- Python **3.8+**
- PyTorch **1.10+**
- CUDA **11.0+** recommandé

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

## Dataset Structure
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
- Images : `.tif` multi-canaux (4 ou 10 canaux)
- Labels : `.tif` masques segmentation
- Multi-class : valeurs 0 → N-1
- Binary : 0 ou 1

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

### Paramètres principaux
- `--model` : unet-alg, unet, unet++, deeplabv3, deeplabv3+, fpn, pspnet, manet, pan, linknet
- `--encoder_name` : resnet34 par défaut
- `--device` : cuda / cpu

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

### Paramètres spécifiques
- `--in_channels`  
- `--use_class_weights`
- `--patch_size`

---

# 🔮 Inference

## 🔹 Inference (Single Patch - Multi-Class)
```bash
python inference_one_patch.py \
  --model_dir /path/to/models \
  --model_name unet++ \
  --image_path /path/to/image.tif \
  --output_dir ./predictions
```

## 🔹 Inference (Single Patch - Binary)
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

### Custom
- **unet-alg**

### SMP Architectures (SMP)
- unet  
- unet++  
- deeplabv3 / deeplabv3+  
- fpn  
- pspnet  
- manet  
- pan  
- linknet  

### Encoders
- ResNet 18–152  
- EfficientNet b0–b7  
- DenseNet, VGG, MobileNet  

---

# 💾 Model Format

### Fichiers générés
- `{model_name}_final_model.pth`
- `{model_name}_best_loss.pth`
- `{model_name}_best_iou.pth`
- `{model_name}_best_combined.pth`
- `{model_name}_metrics.json`
- `{model_name}_training_plot.png`

### Métadonnées incluses
- architecture
- input channels
- nombre de classes
- paramètres d’entraînement
- performances
- géoréférencement

---

# 🚀 Examples

## 1️⃣ Binary Pipeline (Water detection)

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

Inference :
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

### 📉 Small datasets
- `--data_augmentation`
- `--use_class_weights`
- `--pretrained`
- augmenter dropout

### 🗺️ Large images
- patch_size : 256–512
- overlap ≥ 25%

### ⚖️ Binary threshold
- 0.3 = sensible  
- 0.5 = équilibré  
- 0.7 = conservateur  

### 🚀 Performance
- utiliser CUDA  
- réduire batch_size si OOM  
- réduire patch_size en inference  

---

# 🐛 Troubleshooting

### "No images found"
- Vérifier `.tif`
- Structure `images/` et `labels/`

### "Channel mismatch"
```
Expected input channels: 10, received: 4
```
→ ajuster `--in_channels`

### OOM Erreur
- réduire batch_size
- réduire patch_size

### "Model not found"
- Vérifier noms modèles `.pth`

### Artifacts bordures patch
- augmenter `--overlap`

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
Projet sous licence **MIT**.

---

# 🤝 Contributing
1. Fork le repo  
2. Crée une branch  
3. Commit  
4. Push  
5. Pull Request  

---

# 📧 Contact
- Ouvrir une issue GitHub  
- Contacter l’équipe de développement  
