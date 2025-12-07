# SemanticSeg4EO

SemanticSeg4EO 🌍
Semantic Segmentation for Earth Observation
A complete framework for semantic segmentation of satellite images, supporting both binary and multi-class tasks with state-of-the-art architectures.

📋 Table of Contents
Features

Installation

Data Structure

Training

Multi-Class Segmentation

Binary Segmentation

Inference

Single Patch Inference

Large Image Inference

Supported Architectures

Model Format

Examples

Tips and Best Practices

Troubleshooting

✨ Features
🏗️ Model Architectures
UNet-ALG (custom architecture with dropout)

UNet++, DeepLabV3+, FPN, PSPNet, MANet, PAN, LinkNet

Support for ResNet, EfficientNet, etc. encoders

Compatible with segmentation_models_pytorch

🎯 Segmentation Types
Multi-Class Segmentation (up to 6 classes)

Binary Segmentation (1 class, with imbalance handling)

🚀 Advanced Features
Multi-channel data augmentation for satellite imagery

Early stopping with multi-metric monitoring

Automatic checkpoint saving

Georeferencing support (.tif files)

Automatic large image tiling/reconstruction

Advanced metrics (IoU, F1, Precision, Recall)

📥 Installation
Prerequisites
Python 3.8+

PyTorch 1.10+

CUDA 11.0+ (recommended)

Install Dependencies
bash
# Clone the repository
git clone https://github.com/your-username/SemanticSeg4EO.git
cd SemanticSeg4EO

# Install dependencies
pip install -r requirements.txt
Minimum requirements.txt:

text
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
rasterio>=1.2.0
tifffile>=2021.7.2
segmentation-models-pytorch>=0.3.0
matplotlib>=3.4.0
tqdm>=4.62.0
scipy>=1.7.0
📁 Data Structure
For Training
text
dataset_root/
├── Patch/
│   ├── train/
│   │   ├── images/
│   │   │   ├── patch_001.tif
│   │   │   ├── patch_002.tif
│   │   │   └── ...
│   │   └── labels/
│   │       ├── patch_001.tif
│   │       ├── patch_002.tif
│   │       └── ...
│   ├── validation/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
Image Format
Images: Multi-channel .tif files (4 channels for multi-class, 10 channels for Sentinel-2)

Labels: .tif files with segmentation masks

Multi-class: Integer values from 0 to N-1 (N = number of classes)

Binary: 0 (negative) or 1 (positive)

🏋️‍♂️ Training
Multi-Class Segmentation
bash
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
Available parameters:

--dataset_root: Path to dataset root directory (required)

--model: Model architecture (required)

Options: unet-alg, unet, unet++, deeplabv3, deeplabv3+, fpn, pspnet, manet, pan, linknet

--epochs: Number of epochs (default: 100)

--batch_size: Batch size (default: 4)

--save_dir: Directory to save models (default: ./trained_models)

--encoder_name: Encoder name for SMP models (default: resnet34)

--pretrained: Use pretrained encoder weights

--dropout_rate: Dropout rate for UNet-ALG (default: 0.5)

--learning_rate: Learning rate (default: 0.001)

--device: Device to use (default: cuda)

Binary Segmentation
bash
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
Binary-specific parameters:

--in_channels: Number of input channels (default: 10 for Sentinel-2)

--data_augmentation: Enable multi-channel data augmentation

--use_class_weights: Use class weights for imbalance handling

--patch_size: Patch size (default: 224)

🔮 Inference
Single Patch Inference
For Multi-Class Models
bash
python inference_one_patch.py \
  --model_dir /path/to/models \
  --model_name unet++ \
  --image_path /path/to/image.tif \
  --output_dir ./predictions
For Binary Models
bash
python inference_binary.py \
  --model_dir /path/to/models \
  --model_name unet++ \
  --image_path /path/to/image.tif \
  --threshold 0.3 \
  --output_dir ./predictions
Parameters:

--model_dir: Directory containing model files (.pth)

--model_name: Model name (must match training name)

--image_path: Path to input image

--threshold: Threshold for binary segmentation (default: 0.5)

--output_dir: Output directory

Large Image Inference
For Multi-Class Models
bash
python predict_large_image.py \
  --model_path /path/to/model.pth \
  --input /path/to/large_image.tif \
  --output /path/to/prediction.tif \
  --model_name unet++ \
  --patch_size 512 \
  --overlap 128 \
  --device cuda
For Binary Models
bash
python predict_large_image_binary.py \
  --model /path/to/model.pth \
  --input /path/to/large_image.tif \
  --output /path/to/prediction.tif \
  --patch_size 224 \
  --overlap 64 \
  --threshold 0.5
Large image parameters:

--model_path: Exact path to .pth model file

--input: Satellite image to segment

--output: Output path for prediction

--patch_size: Patch size (default: 512)

--overlap: Overlap between patches (default: 128)

--threshold: Threshold for binary segmentation (default: 0.5)

🏗️ Supported Architectures
Custom Architectures
unet-alg: Custom UNet with dropout, optimized for satellite imagery

SMP Architectures (segmentation_models_pytorch)
unet: Classic UNet architecture

unet++: UNet++ with dense connections

deeplabv3 / deeplabv3+: DeepLabV3 and DeepLabV3+

fpn: Feature Pyramid Network

pspnet: Pyramid Scene Parsing Network

manet: Multi-scale Attention Network

pan: Pyramid Attention Network

linknet: LinkNet

Available Encoders
resnet18, resnet34, resnet50, resnet101, resnet152

efficientnet-b0 to efficientnet-b7

mobilenet_v2, densenet121, densenet169, densenet201

vgg11, vgg13, vgg16, vgg19

💾 Model Format
The framework automatically generates multiple model files:

Generated Files
{model_name}_final_model.pth: Final model with complete metadata

{model_name}_best_loss.pth: Best model by validation loss

{model_name}_best_iou.pth: Best model by validation IoU

{model_name}_best_combined.pth: Model with multiple improvements

{model_name}_metrics.json: Detailed training metrics

{model_name}_training_plot.png: Training plots

Included Metadata
Model architecture

Input channels

Number of classes

Training parameters

Performance metrics

Georeferencing information

🚀 Examples
1. Complete Training Pipeline (Binary)
bash
# Train a UNet++ model for water detection
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

# Inference on large image
python predict_large_image_binary.py \
  --model ./models/water_detection/unet++_final_model.pth \
  --input /data/region_complete.tif \
  --output ./predictions/water_mask.tif \
  --patch_size 256 \
  --overlap 64 \
  --threshold 0.3
2. Multi-Class Pipeline
bash
# Train for land cover classification (6 classes)
python main.py \
  --dataset_root /data/landcover \
  --model deeplabv3+ \
  --epochs 200 \
  --batch_size 6 \
  --save_dir ./models/landcover \
  --encoder_name resnet50 \
  --pretrained \
  --learning_rate 0.001

# Predict on a test patch
python inference_one_patch.py \
  --model_dir ./models/landcover \
  --model_name deeplabv3+ \
  --image_path /data/test_patches/patch_884.tif \
  --output_dir ./predictions
💡 Tips and Best Practices
For Small Datasets
Enable --data_augmentation and --use_class_weights

Reduce --batch_size (2-4)

Increase --dropout_rate (0.6-0.7)

Use pretrained encoders (--pretrained)

For Large Images
Patch size: 256-512 pixels

Overlap: 25-50% of patch size

To avoid artifacts: overlap ≥ patch_size // 4

Threshold Selection (Binary)
threshold=0.3: More sensitive (more true positives, risk of false positives)

threshold=0.5: Default balance

threshold=0.7: More conservative (fewer false positives, risk of false negatives)

Performance Optimization
Use --device cuda if available

Adjust --batch_size according to GPU memory

For large image inference, use --patch_size suitable for your GPU

🐛 Troubleshooting
Common Issues
1. "No images found in folder"
Check images are in .tif or .tiff format

Verify folder structure (images/ and labels/)

Ensure image and mask names match

2. "Channel mismatch"
text
Error: Expected input channels: 10, received: 4
Specify --in_channels correctly

Check number of bands in your images

3. "Out of memory" (OOM)
Reduce --batch_size

Reduce --patch_size for inference

Use --device cpu if GPU memory is insufficient

4. Model not found
text
❌ No model found for unet++ in ./models
Verify model name matches exactly

Check .pth files exist in the directory

5. Artifacts on patch borders
Increase --overlap (≥ 64 for patch_size=224)

Try different patch_size and overlap values

Debug Logs
Scripts include detailed logs:

✅ Indicates success

⚠️ Indicates warnings

❌ Indicates errors

📊 Shows statistics and metrics

Example output:

text
🚀 STARTING TRAINING
✅ Dataset loaded: 120 training images
✅ Model unet++ built with 4.2M parameters
🎯 Epoch 1/100: Loss: 0.4521, IoU: 0.6789
💾 Checkpoint best_loss saved
...
🎉 TRAINING COMPLETE: Final IoU: 0.8214
📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

🤝 Contributing
Contributions are welcome! Please:

Fork the project

Create a feature branch

Commit your changes

Push to the branch

Open a Pull Request

📧 Contact
For questions or suggestions:

Create an issue on GitHub

Contact the development team

SemanticSeg4EO - Built for Earth Observation 🌍🛰️
