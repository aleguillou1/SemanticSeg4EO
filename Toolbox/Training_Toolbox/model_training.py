"""
UNIFIED ADVANCED SEGMENTATION TRAINING SYSTEM
-----------------------------------------------------------
Features:
- Supports both Binary and Multi-class segmentation
- Advanced data augmentation for multi-channel imagery
- K-Fold Cross-Validation
- Robust 99th percentile normalization
- Comprehensive callbacks and early stopping
- Dynamic loss and metrics factory
- Compatible with predict_large_image.py
"""

import os
import time
import json
import random
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
import tifffile as tiff
import scipy.stats

# Optional libraries with fallbacks
try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False
    warnings.warn("segmentation_models_pytorch not installed. Only custom models available.")

try:
    from torchvision.models.segmentation import deeplabv3_resnet50
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# =============== CONFIGURATION FACTORY ===============
class SegmentationConfig:
    """Configuration factory for segmentation tasks"""
    
    @staticmethod
    def get_default_config(mode: str = 'binary', num_classes: int = 2):
        """Get default configuration for the segmentation task"""
        config = {
            'mode': mode,
            'num_classes': num_classes if mode == 'multiclass' else 1,
            'task_type': 'binary_segmentation' if mode == 'binary' else 'multiclass_segmentation',
            'normalization': 'percentile_99',
            'mask_dtype': torch.float32 if mode == 'binary' else torch.int64,
            'mask_shape': [1] if mode == 'binary' else [],
            'loss_type': 'dice_bce' if mode == 'binary' else 'dice_ce',
            'activation': 'sigmoid' if mode == 'binary' else 'softmax',
            'metrics_type': 'binary' if mode == 'binary' else 'multiclass'
        }
        return config

# =============== DATA AUGMENTATION ===============
class MultiChannelAugmentation:
    """Advanced data augmentation for multi-channel satellite imagery - GENERIC VERSION"""
    
    def __init__(self, 
                 n_channels: int,
                 augmentation_prob: float = 0.8,
                 geometric_aug: bool = True,
                 noise_aug: bool = True,
                 brightness_aug: bool = True,
                 channel_group_aug: bool = False):
        
        self.n_channels = n_channels
        self.augmentation_prob = augmentation_prob
        self.geometric_aug = geometric_aug
        self.noise_aug = noise_aug
        self.brightness_aug = brightness_aug
        self.channel_group_aug = channel_group_aug
        
        # Dynamic channel grouping (only for Sentinel-2 like data with at least 10 channels)
        self.channel_groups = {}
        if n_channels >= 10 and channel_group_aug:
            # Sentinel-2 band structure (approximate)
            self.channel_groups = {
                'coastal': [0],      # B1 - Coastal aerosol
                'blue': [1],         # B2 - Blue
                'green': [2],        # B3 - Green  
                'red': [3],          # B4 - Red
                'red_edge': [4, 5, 6],  # B5, B6, B7 - Red Edge
                'nir': [7, 8],       # B8, B8A - NIR
                'swir': [9]          # B11 - SWIR
            }
            # Adjust indices if we have fewer channels
            for group_name, indices in list(self.channel_groups.items()):
                self.channel_groups[group_name] = [i for i in indices if i < n_channels]
        
    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.augmentation_prob:
            return img, mask
            
        original_img = img.clone()
        
        try:
            # Apply transformations in random order
            transforms = []
            if self.geometric_aug:
                transforms.extend(['flip_h', 'flip_v', 'rotate'])
            if self.noise_aug:
                transforms.append('noise')
            if self.brightness_aug:
                transforms.append('brightness')
            if self.channel_group_aug and self.channel_groups:
                transforms.append('channel_group')
                
            random.shuffle(transforms)
            
            for transform in transforms:
                if transform == 'flip_h' and random.random() < 0.5:
                    img = torch.flip(img, dims=[2])
                    mask = torch.flip(mask, dims=[2] if mask.dim() == 3 else [1])
                    
                elif transform == 'flip_v' and random.random() < 0.5:
                    img = torch.flip(img, dims=[1])
                    mask = torch.flip(mask, dims=[1] if mask.dim() == 3 else [0])
                    
                elif transform == 'rotate' and random.random() < 0.5:
                    angle = random.choice([0, 90, 180, 270])
                    img = self.rotate_discrete(img, angle)
                    mask = self.rotate_discrete(mask, angle)
                    
                elif transform == 'noise' and random.random() < 0.3:
                    # Correlated noise between channels
                    base_noise = torch.randn(1, img.shape[1], img.shape[2]) * 0.03
                    noise = base_noise.repeat(img.shape[0], 1, 1)
                    img = img + noise
                    
                elif transform == 'brightness' and random.random() < 0.4:
                    # Consistent variation across all channels
                    factor = random.uniform(0.85, 1.15)
                    img = img * factor
                    
                elif transform == 'channel_group' and random.random() < 0.3 and self.channel_groups:
                    # Variation by spectral group
                    group_name = random.choice(list(self.channel_groups.keys()))
                    channel_indices = self.channel_groups[group_name]
                    if channel_indices:  # Check if we have valid indices
                        factor = random.uniform(0.9, 1.1)
                        for idx in channel_indices:
                            if idx < img.shape[0]:
                                img[idx] = img[idx] * factor
            
            # Final clipping
            img = torch.clamp(img, 0, 1)
            
        except Exception as e:
            # In case of error, return original image
            warnings.warn(f"Data augmentation error: {e}")
            return original_img, mask
        
        return img, mask
    
    def rotate_discrete(self, tensor: torch.Tensor, angle: int) -> torch.Tensor:
        """Safe discrete rotation for multi-channels"""
        if angle == 0:
            return tensor
        elif angle == 90:
            return tensor.transpose(1, 2).flip(1)
        elif angle == 180:
            return tensor.flip(1).flip(2)
        elif angle == 270:
            return tensor.transpose(1, 2).flip(2)
        else:
            return tensor

# =============== DATASET ===============
class UnifiedSegmentationDataset(Dataset):
    """Unified dataset for both binary and multi-class segmentation"""
    
    def __init__(self, 
                 folder: str, 
                 image_subdir: str = 'images', 
                 mask_subdir: str = 'labels', 
                 transform: Optional[Callable] = None,
                 mode: str = 'binary',
                 num_classes: int = 2):
        
        self.image_dir = Path(folder) / image_subdir
        self.mask_dir = Path(folder) / mask_subdir
        self.transform = transform
        self.mode = mode
        self.num_classes = num_classes
        
        # Validate mode
        if mode not in ['binary', 'multiclass']:
            raise ValueError(f"Mode must be 'binary' or 'multiclass', got {mode}")
        
        print(f"\nInitializing {mode} dataset:")
        print(f"  - Image dir: {self.image_dir}")
        print(f"  - Mask dir: {self.mask_dir}")
        print(f"  - Num classes: {num_classes}")

        if not self.image_dir.exists():
            raise RuntimeError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise RuntimeError(f"Mask directory not found: {self.mask_dir}")

        valid_extensions = {'.tif', '.tiff', '.TIF', '.TIFF'}
        
        self.images = sorted([
            p for p in self.image_dir.iterdir() 
            if p.is_file() and p.suffix in valid_extensions
        ])
        
        self.masks = sorted([
            p for p in self.mask_dir.iterdir() 
            if p.is_file() and p.suffix in valid_extensions
        ])

        print(f"Found files:")
        print(f"  - Images: {len(self.images)} files")
        print(f"  - Masks: {len(self.masks)} files")

        if len(self.images) == 0:
            available_files = [p.name for p in self.image_dir.iterdir() if p.is_file()]
            raise RuntimeError(
                f"No images found in {self.image_dir}\n"
                f"Valid extensions: {valid_extensions}\n"
                f"Available files: {available_files}"
            )
        
        if len(self.masks) == 0:
            available_files = [p.name for p in self.mask_dir.iterdir() if p.is_file()]
            raise RuntimeError(
                f"No masks found in {self.mask_dir}\n"
                f"Valid extensions: {valid_extensions}\n"
                f"Available files: {available_files}"
            )

        if len(self.images) != len(self.masks):
            warnings.warn(f"Different number of images ({len(self.images)}) and masks ({len(self.masks)})!")

    def __len__(self) -> int:
        return min(len(self.images), len(self.masks))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        
        # Find corresponding mask (same base name)
        img_stem = img_path.stem
        mask_path = None
        
        for mask in self.masks:
            if mask.stem == img_stem:
                mask_path = mask
                break
        
        if mask_path is None:
            raise RuntimeError(f"Corresponding mask not found for {img_path.name}")

        try:
            img = tiff.imread(img_path).astype(np.float32)
            mask = tiff.imread(mask_path)
        except Exception as e:
            raise RuntimeError(f"Error reading file {img_path} or {mask_path}: {e}")

        # Robust normalization (99th percentile)
        if np.max(img) > 0:
            p99 = np.percentile(img, 99)
            img = np.clip(img / p99, 0, 1)
        else:
            img = np.zeros_like(img)

        # Handle mask based on mode
        if self.mode == 'binary':
            # Binary mask: convert to 0/1 and ensure proper shape
            mask = mask.astype(np.float32)
            unique_vals = np.unique(mask)
            if not set(unique_vals).issubset({0, 1}):
                warnings.warn(f"Mask {mask_path.name} contains non-binary values: {unique_vals}")
                mask = (mask > 0).astype(np.float32)
            mask = mask[np.newaxis, ...]  # Add channel dimension: [1, H, W]
        else:
            # Multi-class mask: ensure integer type
            mask = mask.astype(np.int64)
            # Ensure mask values are within [0, num_classes-1]
            mask = np.clip(mask, 0, self.num_classes - 1)

        # Handle image channels
        if img.ndim == 2:
            img = img[..., np.newaxis]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Handle mask based on mode
        if self.mode == 'binary':
            mask = torch.from_numpy(mask).float()  # Shape: [1, H, W]
        else:
            mask = torch.from_numpy(mask).long()   # Shape: [H, W]

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

# =============== LOSS & METRICS FACTORY ===============
class LossFactory:
    """Factory for creating loss functions based on task type"""
    
    @staticmethod
    def create_loss(mode: str, class_weights: Optional[torch.Tensor] = None, 
                    num_classes: int = 2, ignore_index: int = -100):
        """Create appropriate loss function for the task"""
        
        if mode == 'binary':
            return LossFactory._create_binary_loss(class_weights)
        elif mode == 'multiclass':
            return LossFactory._create_multiclass_loss(class_weights, num_classes, ignore_index)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    @staticmethod
    def _create_binary_loss(class_weights: Optional[torch.Tensor] = None):
        """Create binary segmentation loss (Dice + BCE)"""
        
        class DiceBCELoss(nn.Module):
            def __init__(self, weight=None, smooth=1e-6, pos_weight=None):
                super().__init__()
                self.smooth = smooth
                self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
            def forward(self, pred, target):
                bce_loss = self.bce(pred, target)
                
                # Dice loss
                pred_sigmoid = torch.sigmoid(pred)
                intersection = (pred_sigmoid * target).sum(dim=(1, 2, 3))
                union = pred_sigmoid.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
                dice = (2. * intersection + self.smooth) / (union + self.smooth)
                dice_loss = 1 - dice.mean()
                
                return bce_loss + dice_loss
        
        return DiceBCELoss(pos_weight=class_weights)
    
    @staticmethod
    def _create_multiclass_loss(class_weights: Optional[torch.Tensor] = None, 
                                num_classes: int = 6, ignore_index: int = -100):
        """Create multi-class segmentation loss (Dice + CrossEntropy)"""
        
        class DiceCELoss(nn.Module):
            def __init__(self, weight=None, ignore_index=-100):
                super().__init__()
                self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
                
            def forward(self, pred, target):
                ce_loss = self.ce(pred, target)
                
                # Dice loss
                pred_softmax = F.softmax(pred, dim=1)
                if target.dim() == 3:  # [B, H, W]
                    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
                else:  # [B, 1, H, W]
                    target_one_hot = F.one_hot(target.squeeze(1), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
                
                intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
                union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
                dice = (2. * intersection + 1e-6) / (union + 1e-6)
                dice_loss = 1 - dice.mean()
                
                return ce_loss + dice_loss
        
        return DiceCELoss(weight=class_weights, ignore_index=ignore_index)

# =============== METRICS FACTORY - VERSION SMP ===============
class MetricsFactory:
    """Factory for computing metrics using SMP like original code"""
    
    @staticmethod
    def compute_metrics(mode: str, output: torch.Tensor, target: torch.Tensor, 
                        num_classes: int = 2, threshold: float = 0.5):
        """Compute metrics using SMP exactly like original code"""
        if mode == 'binary':
            return MetricsFactory._compute_binary_metrics_smp(output, target, threshold)
        elif mode == 'multiclass':
            return MetricsFactory._compute_multiclass_metrics_smp(output, target, num_classes)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    @staticmethod
    def _compute_binary_metrics_smp(output: torch.Tensor, target: torch.Tensor, 
                                    threshold: float = 0.5):
        """Binary metrics using SMP - EXACTLY like original binary2.py"""
        try:
            # Handle different output shapes like original code
            if output.dim() == 4 and output.shape[1] == 1:
                # Output with channel dimension 1 (typical for binary)
                pred_probs = torch.sigmoid(output)
                pred = (pred_probs > threshold).long().squeeze(1)  # Remove channel dimension
            else:
                # Output already as probabilities
                pred = (output > threshold).long()
            
            # Ensure target has correct dimensions
            if target.dim() == 4 and target.shape[1] == 1:
                target = target.squeeze(1)
            target = target.long()
            
            # Calculate stats with binary mode - EXACTLY like original
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred, 
                target, 
                mode='binary',
                threshold=threshold
            )
            
            # Calculate metrics using SMP - EXACTLY like original
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
            precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
            recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
            
            return {
                'iou': iou_score.item(),
                'f1': f1_score.item(),
                'accuracy': accuracy.item(),
                'precision': precision.item(),
                'recall': recall.item()
            }
                
        except Exception as e:
            print(f"Error in compute_binary_metrics: {e}")
            # Fallback to basic metrics exactly like original
            return MetricsFactory._compute_basic_binary_metrics(output, target, threshold)
    
    @staticmethod
    def _compute_multiclass_metrics_smp(output: torch.Tensor, target: torch.Tensor, 
                                        num_classes: int):
        """Multi-class metrics using SMP - EXACTLY like original model_training.py"""
        try:
            if output.dim() == 4 and target.dim() == 3:
                if not torch.is_tensor(output):
                    output = torch.from_numpy(output)
                if not torch.is_tensor(target):
                    target = torch.from_numpy(target)
                    
                pred = output.argmax(dim=1)
                target = target.long()
                
                # EXACTLY like original code
                tp, fp, fn, tn = smp.metrics.get_stats(
                    pred,
                    target, 
                    mode='multiclass',
                    num_classes=num_classes
                )
                
                # EXACTLY like original code
                iou_score_micro = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                f1_score_micro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                accuracy_micro = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
                
                iou_score_macro = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
                f1_score_macro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
                accuracy_macro = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
                
                return {
                    'iou_micro': iou_score_micro.item(),
                    'f1_micro': f1_score_micro.item(),
                    'accuracy_micro': accuracy_micro.item(),
                    'iou_macro': iou_score_macro.item(),
                    'f1_macro': f1_score_macro.item(),
                    'accuracy_macro': accuracy_macro.item()
                }
            else:
                raise ValueError(f"Invalid dimensions: output {output.dim()}D, target {target.dim()}D")
                
        except Exception as e:
            print(f"Error in compute_metrics: {e}")
            # Fallback exactly like original
            return MetricsFactory._compute_basic_multiclass_metrics(output, target, num_classes)
    
    @staticmethod
    def _compute_basic_binary_metrics(output, target, threshold=0.5):
        """Fallback metrics for binary - EXACTLY like original"""
        # Apply sigmoid if not already done
        pred_probs = torch.sigmoid(output)
        pred = (pred_probs > threshold).float()
        
        # Calculate basic binary metrics
        tp = ((pred == 1) & (target == 1)).sum().float()
        tn = ((pred == 0) & (target == 0)).sum().float()
        fp = ((pred == 1) & (target == 0)).sum().float()
        fn = ((pred == 0) & (target == 1)).sum().float()
        
        # Avoid division by zero
        epsilon = 1e-6
        
        # IoU
        iou = tp / (tp + fp + fn + epsilon)
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        
        # Precision
        precision = tp / (tp + fp + epsilon)
        
        # Recall
        recall = tp / (tp + fn + epsilon)
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        
        return {
            'iou': iou.item(),
            'f1': f1.item(),
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item()
        }
    
    @staticmethod
    def _compute_basic_multiclass_metrics(output, target, num_classes):
        """Fallback metrics for multi-class - EXACTLY like original"""
        pred = output.argmax(dim=1) if output.dim() == 4 else output
        
        iou_list = []
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            
            if union > 0:
                iou = intersection / union
                iou_list.append(iou.item())
            else:
                iou_list.append(0.0)
        
        mean_iou = np.mean(iou_list)
        
        correct = (pred == target).sum().float()
        total = target.numel()
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'iou_micro': mean_iou,
            'f1_micro': mean_iou,
            'accuracy_micro': accuracy.item(),
            'iou_macro': mean_iou,
            'f1_macro': mean_iou,
            'accuracy_macro': accuracy.item()
        }

# =============== MODEL FACTORY ===============
class ModelFactory:
    """Factory for building segmentation models"""
    
    @staticmethod
    def build_model(name: str, in_channels: int = 4, classes: int = 2, 
                    encoder_name: str = 'resnet34', pretrained: bool = False, 
                    dropout_rate: float = 0.5, mode: str = 'binary'):
        """Build segmentation model"""
        
        name = name.lower()
        actual_classes = 1 if mode == 'binary' else classes
        
        # Custom U-Net models
        if name == 'unet-dropout':
            if mode == 'binary':
                return ModelFactory._build_binary_unet(in_channels, dropout_rate)
            else:
                return ModelFactory._build_multiclass_unet(in_channels, actual_classes, dropout_rate)
        
        # SMP models
        if HAS_SMP:
            smp_map = {
                'unet': smp.Unet,
                'unet++': smp.UnetPlusPlus,
                'fpn': smp.FPN,
                'pspnet': smp.PSPNet,
                'linknet': smp.Linknet,
                'manet': smp.MAnet,
                'pan': smp.PAN,
                'deeplabv3': smp.DeepLabV3,
                'deeplabv3+': smp.DeepLabV3Plus
            }
            if name in smp_map:
                # SMP models output logits, we handle activation in loss/metrics
                return smp_map[name](
                    encoder_name=encoder_name,
                    in_channels=in_channels,
                    classes=actual_classes,
                    encoder_weights='imagenet' if pretrained else None,
                    activation=None  # No activation for use with BCEWithLogitsLoss or CrossEntropy
                )
        
        # Torchvision models
        if name == 'deeplabv3' and HAS_TORCHVISION:
            model = deeplabv3_resnet50(pretrained=False, num_classes=actual_classes)
            if in_channels != 3:
                # Adapt first convolution layer
                conv1 = model.backbone.conv1
                new_conv = nn.Conv2d(in_channels, conv1.out_channels, 
                                   kernel_size=conv1.kernel_size, 
                                   stride=conv1.stride, padding=conv1.padding,
                                   bias=conv1.bias is not None)
                with torch.no_grad():
                    for i in range(min(in_channels, 3)):
                        new_conv.weight[:, i, :, :] = conv1.weight[:, i, :, :]
                    if in_channels > 3:
                        for j in range(3, in_channels):
                            new_conv.weight[:, j, :, :] = conv1.weight[:, 0, :, :]
                model.backbone.conv1 = new_conv
            return model
        
        raise ValueError(f"Model {name} not supported. Install segmentation_models_pytorch or use 'unet-dropout'.")
    
    @staticmethod
    def _build_binary_unet(in_channels: int, dropout_rate: float = 0.5):
        """Build binary U-Net with dropout"""
        
        class DoubleConv(nn.Module):
            def __init__(self, in_ch, out_ch, dropout_rate=0.5):
                super().__init__()
                self.double_conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout_rate),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            def forward(self, x):
                return self.double_conv(x)
        
        class BinaryUNet(nn.Module):
            def __init__(self, in_channels=10, out_channels=1, dropout_rate=0.5):
                super().__init__()
                
                # Encoder
                self.enc1 = DoubleConv(in_channels, 32, dropout_rate)
                self.pool1 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout2d(dropout_rate))
                
                self.enc2 = DoubleConv(32, 64, dropout_rate)
                self.pool2 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout2d(dropout_rate))
                
                self.enc3 = DoubleConv(64, 128, dropout_rate)
                self.pool3 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout2d(dropout_rate))
                
                # Center
                self.center = DoubleConv(128, 256, dropout_rate)
                
                # Decoder
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
                self.dec3 = DoubleConv(256 + 128, 128, dropout_rate)
                self.dec2 = DoubleConv(128 + 64, 64, dropout_rate)
                self.dec1 = DoubleConv(64 + 32, 32, dropout_rate)
                
                # Output logits (no sigmoid)
                self.final = nn.Conv2d(32, out_channels, kernel_size=1)
            
            def forward(self, x):
                # Encoder
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool1(e1))
                e3 = self.enc3(self.pool2(e2))
                
                # Center
                c = self.center(self.pool3(e3))
                
                # Decoder
                d3 = self.up(c)
                d3 = torch.cat([d3, e3], dim=1)
                d3 = self.dec3(d3)
                
                d2 = self.up(d3)
                d2 = torch.cat([d2, e2], dim=1)
                d2 = self.dec2(d2)
                
                d1 = self.up(d2)
                d1 = torch.cat([d1, e1], dim=1)
                d1 = self.dec1(d1)
                
                return self.final(d1)
        
        return BinaryUNet(in_channels, 1, dropout_rate)
    
    @staticmethod
    def _build_multiclass_unet(in_channels: int, num_classes: int, dropout_rate: float = 0.5):
        """Build multi-class U-Net with dropout"""
        
        class DoubleConv(nn.Module):
            def __init__(self, in_ch, out_ch, dropout_rate=0.5):
                super().__init__()
                self.double_conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout_rate),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            def forward(self, x):
                return self.double_conv(x)
        
        class MultiClassUNet(nn.Module):
            def __init__(self, in_channels=4, out_channels=6, dropout_rate=0.5):
                super().__init__()
                
                # Encoder
                self.enc1 = DoubleConv(in_channels, 64, dropout_rate)
                self.pool1 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout2d(dropout_rate))
                
                self.enc2 = DoubleConv(64, 128, dropout_rate)
                self.pool2 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout2d(dropout_rate))
                
                self.enc3 = DoubleConv(128, 256, dropout_rate)
                self.pool3 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout2d(dropout_rate))
                
                self.enc4 = DoubleConv(256, 512, dropout_rate)
                self.pool4 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout2d(dropout_rate))
                
                # Center
                self.center = DoubleConv(512, 1024, dropout_rate)
                
                # Decoder
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
                self.dec4 = DoubleConv(1024 + 512, 512, dropout_rate)
                self.dec3 = DoubleConv(512 + 256, 256, dropout_rate)
                self.dec2 = DoubleConv(256 + 128, 128, dropout_rate)
                self.dec1 = DoubleConv(128 + 64, 64, dropout_rate)
                
                # Output logits (no softmax)
                self.final = nn.Conv2d(64, out_channels, kernel_size=1)
            
            def forward(self, x):
                # Encoder
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool1(e1))
                e3 = self.enc3(self.pool2(e2))
                e4 = self.enc4(self.pool3(e3))
                
                # Center
                c = self.center(self.pool4(e4))
                
                # Decoder
                d4 = self.up(c)
                d4 = torch.cat([d4, e4], dim=1)
                d4 = self.dec4(d4)
                
                d3 = self.up(d4)
                d3 = torch.cat([d3, e3], dim=1)
                d3 = self.dec3(d3)
                
                d2 = self.up(d3)
                d2 = torch.cat([d2, e2], dim=1)
                d2 = self.dec2(d2)
                
                d1 = self.up(d2)
                d1 = torch.cat([d1, e1], dim=1)
                d1 = self.dec1(d1)
                
                return self.final(d1)
        
        return MultiClassUNet(in_channels, num_classes, dropout_rate)

# =============== TRAINING CALLBACKS ===============
class TrainingCallbacks:
    """Advanced training callbacks with early stopping and checkpointing"""
    
    def __init__(self, save_dir: str, model_name: str, patience: int = 15, 
                 min_delta: float = 1e-4, mode: str = 'binary'):
        self.save_dir = save_dir
        self.model_name = model_name
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_loss = float('inf')
        self.best_iou = 0.0
        self.best_f1 = 0.0
        self.counter = 0
        self.early_stop = False
        
    def on_epoch_end(self, epoch: int, model: nn.Module, val_loss: float, 
                     val_iou: float, val_f1: float, optimizer: torch.optim.Optimizer,
                     scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """Callback at end of each epoch"""
        improvement_found = False
        loss_improved = False
        iou_improved = False
        f1_improved = False
        
        # Save best loss model
        if val_loss < self.best_loss - self.min_delta:
            print(f"New best loss: {val_loss:.4f} (previous: {self.best_loss:.4f})")
            self.best_loss = val_loss
            improvement_found = True
            loss_improved = True
            
            checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_best_loss.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_f1': val_f1,
                'best_loss': self.best_loss,
                'mode': self.mode
            }, checkpoint_path)
            print(f"Saved best_loss checkpoint")
            
        # Save best IoU model
        if val_iou > self.best_iou + self.min_delta:
            metric_name = 'IoU' if self.mode == 'binary' else 'mIoU'
            print(f"New best {metric_name}: {val_iou:.4f} (previous: {self.best_iou:.4f})")
            self.best_iou = val_iou
            improvement_found = True
            iou_improved = True
            
            checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_best_iou.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_f1': val_f1,
                'best_iou': self.best_iou,
                'mode': self.mode
            }, checkpoint_path)
            print(f"Saved best_iou checkpoint")
        
        # Check F1 improvement
        if val_f1 > self.best_f1 + self.min_delta:
            print(f"New best F1: {val_f1:.4f} (previous: {self.best_f1:.4f})")
            self.best_f1 = val_f1
            improvement_found = True
            f1_improved = True
        
        # Save combined model if any metric improved
        if improvement_found:
            self.counter = 0
            
            checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_best_combined.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_f1': val_f1,
                'best_loss': self.best_loss,
                'best_iou': self.best_iou,
                'best_f1': self.best_f1,
                'improvements': {
                    'loss_improved': loss_improved,
                    'iou_improved': iou_improved,
                    'f1_improved': f1_improved
                },
                'mode': self.mode
            }, checkpoint_path)
            print(f"Saved combined checkpoint (improvements: loss={loss_improved}, iou={iou_improved}, f1={f1_improved})")
            
        else:
            # No improvement on main metrics
            self.counter += 1
            print(f"No significant improvement - Patience: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered after {epoch+1} epochs")
                
        return self.early_stop

# =============== UTILITIES ===============
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model: nn.Module, device: torch.device, 
                          input_size: Tuple[int, int, int, int] = (1, 4, 224, 224), 
                          n_runs: int = 50) -> float:
    """Measure inference time"""
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / n_runs
    return avg_time

def get_model_complexity(model: nn.Module, device: torch.device, 
                        input_size: Tuple[int, int, int, int] = (1, 4, 224, 224)) -> Dict:
    """Get model complexity metrics"""
    print("\nModel complexity analysis:")
    
    params = count_parameters(model)
    print(f"  - Parameters: {params:,}")
    
    inf_time = measure_inference_time(model, device, input_size=input_size)
    print(f"  - Inference time: {inf_time:.4f}s")
    
    return {
        'parameters': params,
        'inference_time': inf_time
    }

def save_model(model: nn.Module, model_path: str, metadata: Optional[Dict] = None):
    """Save model in standardized format"""
    model_info = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'model_architecture': str(model),
        'metadata': metadata or {}
    }
    torch.save(model_info, model_path)
    print(f"Model saved: {model_path}")

def calculate_class_weights(dataloader: DataLoader, device: torch.device, 
                           mode: str = 'binary', num_classes: int = 2) -> Optional[torch.Tensor]:
    """Calculate class weights to handle imbalance"""
    
    if mode == 'binary':
        total_pixels = 0
        positive_pixels = 0
        
        for _, masks in dataloader:
            positive_pixels += (masks > 0).sum().item()
            total_pixels += masks.numel()
        
        negative_pixels = total_pixels - positive_pixels
        positive_weight = negative_pixels / (positive_pixels + 1e-6)
        
        print(f"Class statistics (binary):")
        print(f"  - Positive pixels: {positive_pixels} ({positive_pixels/total_pixels*100:.2f}%)")
        print(f"  - Negative pixels: {negative_pixels} ({negative_pixels/total_pixels*100:.2f}%)")
        print(f"  - Positive class weight: {positive_weight:.2f}")
        
        return torch.tensor([positive_weight], device=device)
    
    else:  # Multi-class
        class_counts = torch.zeros(num_classes, device=device)
        total_pixels = 0
        
        for _, masks in dataloader:
            masks = masks.to(device)
            for cls in range(num_classes):
                class_counts[cls] += (masks == cls).sum().item()
            total_pixels += masks.numel()
        
        # Inverse frequency weighting
        class_weights = total_pixels / (num_classes * class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
        
        print(f"Class statistics (multi-class):")
        for cls in range(num_classes):
            percentage = class_counts[cls] / total_pixels * 100
            print(f"  - Class {cls}: {class_counts[cls]:.0f} pixels ({percentage:.2f}%), weight: {class_weights[cls]:.2f}")
        
        return class_weights

# =============== EVALUATION ===============
def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device,
                  mode: str = 'binary', num_classes: int = 2, threshold: float = 0.5) -> Dict:
    """Standard evaluation with metrics"""
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            
            if isinstance(out, dict):  # Handle torchvision models
                out = out['out']
                
            all_outputs.append(out.cpu())
            all_targets.append(masks.cpu())
    
    outputs = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    metrics = MetricsFactory.compute_metrics(mode, outputs, targets, num_classes, threshold)
    return metrics

def evaluate_model_with_loss(model: nn.Module, dataloader: DataLoader, device: torch.device,
                            criterion: nn.Module, mode: str = 'binary', 
                            num_classes: int = 2, threshold: float = 0.5) -> Dict:
    """Evaluation with loss and metrics"""
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            
            if isinstance(out, dict):
                out = out['out']
                
            loss = criterion(out, masks)
            total_loss += loss.item() * imgs.size(0)
            
            all_outputs.append(out.cpu())
            all_targets.append(masks.cpu())
    
    mean_loss = total_loss / len(dataloader.dataset)
    
    outputs = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = MetricsFactory.compute_metrics(mode, outputs, targets, num_classes, threshold)
    
    result = {
        'loss': mean_loss,
        'detailed_metrics': metrics
    }
    
    # Add appropriate main metrics based on mode
    if mode == 'binary':
        result.update({
            'mean_iou': metrics['iou'],
            'mean_f1': metrics['f1'],
            'accuracy': metrics['accuracy']
        })
    else:
        result.update({
            'mean_iou': metrics['mean_iou'] if 'mean_iou' in metrics else metrics['iou_macro'],
            'mean_f1': metrics['f1_macro'],
            'accuracy': metrics['accuracy_macro']
        })
    
    return result

# =============== TRAINING ===============
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               device: torch.device, mode: str = 'binary', num_classes: int = 2,
               epochs: int = 100, lr: float = 1e-3, save_dir: str = './saved_models',
               model_name: str = 'model', class_weight: Optional[torch.Tensor] = None):
    """Main training loop for segmentation models"""
    
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Loss function
    criterion = LossFactory.create_loss(mode, class_weight, num_classes)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Callbacks
    callbacks = TrainingCallbacks(save_dir, model_name, patience=15, min_delta=1e-4, mode=mode)
    
    # History tracking
    history = {
        'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': [], 
        'val_accuracy': [], 'epoch_time': [], 'learning_rate': []
    }
    
    # Add mode-specific history
    if mode == 'binary':
        history.update({'val_precision': [], 'val_recall': []})
    else:
        history.update({'val_iou_micro': [], 'val_iou_macro': []})
    
    for epoch in range(epochs):
        model.train()
        train_loss, start_time = 0, time.perf_counter()
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        val_metrics = evaluate_model_with_loss(model, val_loader, device, criterion, mode, num_classes)
        
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_iou'].append(val_metrics['mean_iou'])
        history['val_f1'].append(val_metrics['mean_f1'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['epoch_time'].append(time.perf_counter() - start_time)
        history['learning_rate'].append(current_lr)
        
        # Mode-specific history
        if mode == 'binary':
            history['val_precision'].append(val_metrics['detailed_metrics']['precision'])
            history['val_recall'].append(val_metrics['detailed_metrics']['recall'])
        else:
            history['val_iou_micro'].append(val_metrics['detailed_metrics']['iou_micro'])
            history['val_iou_macro'].append(val_metrics['detailed_metrics']['iou_macro'])
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Loss: train={train_loss:.4f}, val={val_metrics['loss']:.4f}")
        print(f"  {'IoU' if mode == 'binary' else 'mIoU'}: {val_metrics['mean_iou']:.4f}, F1: {val_metrics['mean_f1']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        if mode == 'binary':
            print(f"  Precision: {val_metrics['detailed_metrics']['precision']:.4f}, Recall: {val_metrics['detailed_metrics']['recall']:.4f}")
        print(f"  LR: {current_lr:.2e}, Time: {history['epoch_time'][-1]:.1f}s")
        
        # Callbacks
        early_stop = callbacks.on_epoch_end(
            epoch, model, val_metrics['loss'], val_metrics['mean_iou'], 
            val_metrics['mean_f1'], optimizer, scheduler
        )
        
        if early_stop:
            print("Early stopping")
            break
    
    return model, history, callbacks.best_loss, callbacks.best_iou

# =============== VISUALIZATION ===============
def plot_training_history(history: Dict, save_path: str, mode: str = 'binary'):
    """Plot training history"""
    
    if mode == 'binary':
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax1.plot(history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # IoU
        ax2.plot(history['val_iou'], 'g-', label='Validation IoU')
        ax2.set_title('Validation IoU')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU')
        ax2.legend()
        ax2.grid(True)
        
        # F1 Score and Accuracy
        ax3.plot(history['val_f1'], 'r-', label='Validation F1 Score')
        ax3.plot(history['val_accuracy'], 'b-', label='Validation Accuracy')
        ax3.set_title('Validation F1 & Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True)
        
        # Precision and Recall
        ax4.plot(history['val_precision'], 'c-', label='Validation Precision')
        ax4.plot(history['val_recall'], 'm-', label='Validation Recall')
        ax4.set_title('Validation Precision & Recall')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True)
        
    else:  # Multi-class
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax1.plot(history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # mIoU
        ax2.plot(history['val_iou'], 'g-', label='Validation mIoU')
        if 'val_iou_macro' in history:
            ax2.plot(history['val_iou_macro'], 'g--', label='Validation mIoU (macro)')
        if 'val_iou_micro' in history:
            ax2.plot(history['val_iou_micro'], 'g:', label='Validation mIoU (micro)')
        ax2.set_title('Validation mIoU')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mIoU')
        ax2.legend()
        ax2.grid(True)
        
        # F1 Score and Accuracy
        ax3.plot(history['val_f1'], 'r-', label='Validation F1 Score')
        ax3.plot(history['val_accuracy'], 'b-', label='Validation Accuracy')
        ax3.set_title('Validation F1 & Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True)
        
        # Learning Rate
        ax4.plot(history['learning_rate'], 'm-', label='Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# =============== CROSS-VALIDATION ===============
class KFoldCrossValidator:
    """K-Fold Cross Validation for segmentation"""
    
    def __init__(self, dataset_root: str, patch_subdir: str = 'Patch', 
                 subdirectories: List[str] = None, n_splits: int = 5, 
                 random_state: int = 42, data_augmentation: bool = False, 
                 in_channels: int = 10, mode: str = 'binary', num_classes: int = 2):
        
        self.dataset_root = Path(dataset_root)
        self.patch_subdir = patch_subdir
        self.subdirectories = subdirectories or ['train', 'validation', 'test']
        self.n_splits = n_splits
        self.random_state = random_state
        self.data_augmentation = data_augmentation
        self.in_channels = in_channels
        self.mode = mode
        self.num_classes = num_classes
        self.folds = []
        
    def prepare_folds(self):
        """Prepare K-Fold splits from the dataset"""
        all_pairs = []
        
        for subdir in self.subdirectories:
            image_dir = self.dataset_root / self.patch_subdir / subdir / 'images'
            mask_dir = self.dataset_root / self.patch_subdir / subdir / 'labels'
            
            if not image_dir.exists() or not mask_dir.exists():
                warnings.warn(f"Directory not found: {image_dir} or {mask_dir}, skipping...")
                continue
            
            valid_extensions = {'.tif', '.tiff', '.TIF', '.TIFF'}
            
            images = sorted([p for p in image_dir.iterdir() 
                           if p.is_file() and p.suffix in valid_extensions])
            
            masks = sorted([p for p in mask_dir.iterdir() 
                          if p.is_file() and p.suffix in valid_extensions])
            
            # Create pairs
            for img_path in images:
                img_stem = img_path.stem
                mask_path = None
                
                for mask in masks:
                    if mask.stem == img_stem:
                        mask_path = mask
                        break
                
                if mask_path is None:
                    warnings.warn(f"Corresponding mask not found for {img_path.name}, skipping...")
                    continue
                
                all_pairs.append((img_path, mask_path))
        
        if len(all_pairs) == 0:
            raise RuntimeError(f"No valid image-mask pairs found.")
        
        print(f"\nTotal pairs collected: {len(all_pairs)}")
        
        # Create K-Fold splits
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_pairs)):
            train_pairs = [all_pairs[i] for i in train_idx]
            val_pairs = [all_pairs[i] for i in val_idx]
            
            self.folds.append({
                'fold': fold_idx,
                'train_pairs': train_pairs,
                'val_pairs': val_pairs,
                'train_size': len(train_pairs),
                'val_size': len(val_pairs)
            })
        
        print(f"\nPrepared {self.n_splits}-fold cross-validation:")
        for fold in self.folds:
            print(f"  Fold {fold['fold']}: {fold['train_size']} train, {fold['val_size']} validation")
    
    def get_fold_datasets(self, fold_idx: int, transform: Optional[Callable] = None):
        """Get datasets for a specific fold"""
        if fold_idx >= len(self.folds):
            raise ValueError(f"Fold {fold_idx} not found.")
        
        fold = self.folds[fold_idx]
        
        # Create custom datasets
        train_dataset = _DatasetFromPairs(
            fold['train_pairs'], transform=transform, 
            mode=self.mode, num_classes=self.num_classes
        )
        val_dataset = _DatasetFromPairs(
            fold['val_pairs'], transform=None,  # No augmentation for validation
            mode=self.mode, num_classes=self.num_classes
        )
        
        return train_dataset, val_dataset

class _DatasetFromPairs(Dataset):
    """Internal dataset from explicit pairs (for cross-validation)"""
    
    def __init__(self, pairs: List[Tuple[Path, Path]], transform: Optional[Callable] = None,
                 mode: str = 'binary', num_classes: int = 2):
        self.pairs = pairs
        self.transform = transform
        self.mode = mode
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        try:
            img = tiff.imread(img_path).astype(np.float32)
            mask = tiff.imread(mask_path)
        except Exception as e:
            raise RuntimeError(f"Error reading file {img_path} or {mask_path}: {e}")
        
        # Robust normalization
        if np.max(img) > 0:
            p99 = np.percentile(img, 99)
            img = np.clip(img / p99, 0, 1)
        
        # Handle mask based on mode
        if self.mode == 'binary':
            mask = mask.astype(np.float32)
            if not set(np.unique(mask)).issubset({0, 1}):
                mask = (mask > 0).astype(np.float32)
            mask = mask[np.newaxis, ...]
        else:
            mask = mask.astype(np.int64)
            mask = np.clip(mask, 0, self.num_classes - 1)
        
        # Handle image channels
        if img.ndim == 2:
            img = img[..., np.newaxis]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Handle mask
        if self.mode == 'binary':
            mask = torch.from_numpy(mask).float()
        else:
            mask = torch.from_numpy(mask).long()
        
        if self.transform:
            img, mask = self.transform(img, mask)
        
        return img, mask

def cross_validate_model(model_name: str, dataset_root: str, device: str = 'cuda',
                        n_splits: int = 5, batch_size: int = 4, epochs: int = 100,
                        save_dir: str = './saved_models_cv', encoder_name: str = 'resnet34',
                        pretrained: bool = False, dropout_rate: float = 0.5,
                        learning_rate: float = 1e-3, in_channels: int = 10,
                        patch_size: int = 224, data_augmentation: bool = False,
                        use_class_weights: bool = True, patch_subdir: str = 'Patch',
                        subdirectories: List[str] = None, mode: str = 'binary',
                        num_classes: int = 2, val_strategy: str = 'split') -> Dict:
    """Perform K-Fold Cross Validation"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"K-FOLD CROSS VALIDATION (n_splits={n_splits}, mode={mode})")
    print(f"{'='*60}")
    
    # Create save directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    cv_save_dir = Path(save_dir) / f"cv_{model_name}_{mode}_{timestamp}"
    cv_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare cross-validator
    cv = KFoldCrossValidator(
        dataset_root=dataset_root,
        patch_subdir=patch_subdir,
        subdirectories=subdirectories or ['train', 'validation', 'test'],
        n_splits=n_splits,
        data_augmentation=data_augmentation,
        in_channels=in_channels,
        mode=mode,
        num_classes=num_classes
    )
    cv.prepare_folds()
    
    # Data augmentation
    if data_augmentation:
        augmentation = MultiChannelAugmentation(
            n_channels=in_channels,
            augmentation_prob=0.7,
            geometric_aug=True,
            noise_aug=True,
            brightness_aug=True,
            channel_group_aug=(in_channels >= 10)  # Only for Sentinel-2 like data
        )
    else:
        augmentation = None
    
    fold_results = []
    fold_histories = []
    
    # Train each fold
    for fold_idx in range(n_splits):
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx + 1}/{n_splits} ({mode})")
        print(f"{'='*60}")
        
        # Get datasets for this fold
        train_dataset, val_dataset = cv.get_fold_datasets(fold_idx, transform=augmentation)
        
        # Create data loaders
        effective_batch_size = min(batch_size, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, 
                                  shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, 
                                shuffle=False, pin_memory=True)
        
        # Calculate class weights
        class_weight = None
        if use_class_weights:
            class_weight = calculate_class_weights(train_loader, device, mode, num_classes)
        
        # Build model
        model = ModelFactory.build_model(
            model_name, in_channels=in_channels, classes=num_classes,
            encoder_name=encoder_name, pretrained=pretrained,
            dropout_rate=dropout_rate, mode=mode
        )
        
        # Adjusted learning rate for small datasets
        adjusted_lr = learning_rate * 0.5
        
        print(f"\nFold {fold_idx + 1} Configuration:")
        print(f"  - Mode: {mode}")
        print(f"  - Train size: {len(train_dataset)}")
        print(f"  - Validation size: {len(val_dataset)}")
        print(f"  - Batch size: {effective_batch_size}")
        print(f"  - Learning rate: {adjusted_lr:.1e}")
        if class_weight is not None:
            print(f"  - Class weights: Enabled")
        
        # Create fold-specific save directory
        fold_save_dir = cv_save_dir / f"fold_{fold_idx}"
        fold_save_dir.mkdir(exist_ok=True)
        
        # Train model for this fold
        model, history, best_loss, best_iou = train_model(
            model, train_loader, val_loader, device, mode, num_classes,
            epochs=epochs, lr=adjusted_lr, save_dir=str(fold_save_dir),
            model_name=f"{model_name}_fold{fold_idx}", class_weight=class_weight
        )
        
        # Evaluate on validation set
        criterion = LossFactory.create_loss(mode, class_weight, num_classes)
        val_metrics = evaluate_model_with_loss(model, val_loader, device, criterion, mode, num_classes)
        
        # Save fold results
        fold_result = {
            'fold': fold_idx,
            'mode': mode,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'best_val_loss': best_loss,
            'best_val_iou': best_iou,
            'final_val_metrics': val_metrics,
            'history': {
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'val_iou': history['val_iou'],
                'val_f1': history['val_f1']
            }
        }
        
        fold_results.append(fold_result)
        fold_histories.append(history)
        
        # Save fold model
        model_path = fold_save_dir / f"{model_name}_fold{fold_idx}_final.pth"
        metadata = {
            'model_name': model_name,
            'fold': fold_idx,
            'mode': mode,
            'in_channels': in_channels,
            'num_classes': num_classes,
            'encoder_name': encoder_name,
            'fold_metrics': fold_result
        }
        save_model(model, str(model_path), metadata)
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Best Val Loss: {best_loss:.4f}")
        print(f"  Best Val IoU: {best_iou:.4f}")
        print(f"  Final Val IoU: {val_metrics['mean_iou']:.4f}")
        print(f"  Final Val F1: {val_metrics['mean_f1']:.4f}")
        print(f"  Final Val Accuracy: {val_metrics['accuracy']:.4f}")
    
    # Calculate cross-validation statistics
    cv_stats = _calculate_cv_statistics(fold_results, mode)
    
    # Plot results
    _plot_cross_validation_results(fold_results, fold_histories, cv_save_dir, model_name, mode)
    
    # Save report
    _save_cv_report(fold_results, cv_stats, cv_save_dir, model_name, mode)
    
    return {
        'fold_results': fold_results,
        'cv_stats': cv_stats,
        'cv_save_dir': str(cv_save_dir),
        'mode': mode,
        'num_classes': num_classes
    }

def _calculate_cv_statistics(fold_results: List[Dict], mode: str) -> Dict:
    """Calculate statistics across all folds"""
    if mode == 'binary':
        ious = [fold['final_val_metrics']['mean_iou'] for fold in fold_results]
        f1s = [fold['final_val_metrics']['mean_f1'] for fold in fold_results]
    else:
        ious = [fold['final_val_metrics']['mean_iou'] for fold in fold_results]
        f1s = [fold['final_val_metrics']['mean_f1'] for fold in fold_results]
    
    n_folds = len(fold_results)
    t_value = scipy.stats.t.ppf(0.975, n_folds - 1)
    
    stats = {
        'mean_iou': np.mean(ious),
        'std_iou': np.std(ious),
        'mean_f1': np.mean(f1s),
        'std_f1': np.std(f1s),
        'fold_ious': ious,
        'fold_f1s': f1s,
        'ci_iou': {
            'lower': np.mean(ious) - t_value * (np.std(ious) / np.sqrt(n_folds)),
            'upper': np.mean(ious) + t_value * (np.std(ious) / np.sqrt(n_folds))
        },
        'ci_f1': {
            'lower': np.mean(f1s) - t_value * (np.std(f1s) / np.sqrt(n_folds)),
            'upper': np.mean(f1s) + t_value * (np.std(f1s) / np.sqrt(n_folds))
        }
    }
    
    return stats

def _plot_cross_validation_results(fold_results: List[Dict], fold_histories: List[Dict],
                                  save_dir: Path, model_name: str, mode: str):
    """Plot cross-validation results"""
    
    # Plot 1: Validation IoU across folds
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot fold IoU trajectories
    for fold_idx, history in enumerate(fold_histories):
        axes[0, 0].plot(history['val_iou'], label=f'Fold {fold_idx+1}')
    axes[0, 0].set_title(f'Validation {"IoU" if mode == "binary" else "mIoU"} across Folds')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('IoU' if mode == 'binary' else 'mIoU')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot fold F1 trajectories
    for fold_idx, history in enumerate(fold_histories):
        axes[0, 1].plot(history['val_f1'], label=f'Fold {fold_idx+1}')
    axes[0, 1].set_title('Validation F1 across Folds')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot fold losses
    for fold_idx, history in enumerate(fold_histories):
        axes[0, 2].plot(history['val_loss'], label=f'Fold {fold_idx+1}')
    axes[0, 2].set_title('Validation Loss across Folds')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot final metrics comparison
    final_ious = [fold['final_val_metrics']['mean_iou'] for fold in fold_results]
    final_f1s = [fold['final_val_metrics']['mean_f1'] for fold in fold_results]
    
    axes[1, 0].bar(range(len(final_ious)), final_ious)
    axes[1, 0].set_title(f'Final {"IoU" if mode == "binary" else "mIoU"} per Fold')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('IoU' if mode == 'binary' else 'mIoU')
    axes[1, 0].axhline(y=np.mean(final_ious), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(final_ious):.3f}')
    axes[1, 0].legend()
    
    axes[1, 1].bar(range(len(final_f1s)), final_f1s)
    axes[1, 1].set_title('Final F1 per Fold')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].axhline(y=np.mean(final_f1s), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(final_f1s):.3f}')
    axes[1, 1].legend()
    
    # Plot distribution of metrics
    axes[1, 2].boxplot([final_ious, final_f1s], labels=['IoU', 'F1'])
    axes[1, 2].set_title('Distribution of Metrics across Folds')
    axes[1, 2].set_ylabel('Score')
    
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_cv_results_{mode}.png", dpi=300, bbox_inches='tight')
    plt.close()

def _save_cv_report(fold_results: List[Dict], cv_stats: Dict, save_dir: Path, 
                   model_name: str, mode: str):
    """Save comprehensive cross-validation report"""
    
    report = {
        'model_name': model_name,
        'mode': mode,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_folds': len(fold_results),
        'fold_details': fold_results,
        'cv_statistics': cv_stats,
        'summary': {
            'mean_iou': cv_stats['mean_iou'],
            'std_iou': cv_stats['std_iou'],
            'mean_f1': cv_stats['mean_f1'],
            'std_f1': cv_stats['std_f1'],
            'cv_stability': 'High' if cv_stats['std_iou'] < 0.05 else 
                           'Medium' if cv_stats['std_iou'] < 0.1 else 'Low',
            'performance_level': 'Excellent' if cv_stats['mean_iou'] > 0.7 else 
                                'Good' if cv_stats['mean_iou'] > 0.5 else 'Poor'
        }
    }
    
    report_path = save_dir / f"{model_name}_cv_report_{mode}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

# =============== MAIN TRAINING FUNCTION ===============
def train_and_save_model(model_name: str, dataset_root: str, mode: str = 'binary',
                        num_classes: int = 2, device: str = 'cuda', batch_size: int = 4,
                        epochs: int = 100, save_dir: str = './saved_models',
                        encoder_name: str = 'resnet34', pretrained: bool = False,
                        dropout_rate: float = 0.5, learning_rate: float = 1e-3,
                        in_channels: int = 10, patch_size: int = 224,
                        data_augmentation: bool = False, use_class_weights: bool = True,
                        patch_subdir: str = 'Patch', val_strategy: str = 'split') -> Dict:
    """Main training function for unified segmentation"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"TRAINING {mode.upper()} SEGMENTATION MODEL")
    print(f"{'='*60}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Data augmentation
    if data_augmentation:
        augmentation = MultiChannelAugmentation(
            n_channels=in_channels,
            augmentation_prob=0.7,
            geometric_aug=True,
            noise_aug=True,
            brightness_aug=True,
            channel_group_aug=(in_channels >= 10)
        )
        print(f"Multi-channel data augmentation enabled")
    else:
        augmentation = None
        print(f"Data augmentation disabled")
    
    # Load datasets
    train_path = os.path.join(dataset_root, patch_subdir, 'train')
    val_path = os.path.join(dataset_root, patch_subdir, 'validation')
    test_path = os.path.join(dataset_root, patch_subdir, 'test')
    
    train_dataset = UnifiedSegmentationDataset(train_path, transform=augmentation, 
                                              mode=mode, num_classes=num_classes)
    val_dataset = UnifiedSegmentationDataset(val_path, mode=mode, num_classes=num_classes)
    test_dataset = UnifiedSegmentationDataset(test_path, mode=mode, num_classes=num_classes)
    
    # Adjust batch size
    effective_batch_size = min(batch_size, len(train_dataset))
    if effective_batch_size != batch_size:
        print(f"Batch_size adjusted from {batch_size} to {effective_batch_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Calculate class weights
    class_weight = None
    if use_class_weights:
        class_weight = calculate_class_weights(train_loader, device, mode, num_classes)
    
    # Build model
    print(f"\n=== Building {model_name} ({mode}, {num_classes} classes) ===")
    model = ModelFactory.build_model(
        model_name, in_channels=in_channels, classes=num_classes,
        encoder_name=encoder_name, pretrained=pretrained,
        dropout_rate=dropout_rate, mode=mode
    )
    
    # Adjusted learning rate
    adjusted_lr = learning_rate * 0.5
    
    print(f"\nConfiguration:")
    print(f"  - Mode: {mode}")
    print(f"  - Input channels: {in_channels}")
    print(f"  - Num classes: {num_classes}")
    print(f"  - Dataset size: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"  - Effective batch size: {effective_batch_size}")
    print(f"  - Learning rate: {adjusted_lr:.1e}")
    print(f"  - Data augmentation: {'Enabled' if data_augmentation else 'Disabled'}")
    print(f"  - Class weights: {'Enabled' if use_class_weights else 'Disabled'}")
    
    # Compute model complexity
    input_size = (1, in_channels, patch_size, patch_size)
    complexity_info = get_model_complexity(model, device, input_size=input_size)
    
    # Train model
    model, history, best_loss, best_iou = train_model(
        model, train_loader, val_loader, device, mode, num_classes,
        epochs=epochs, lr=adjusted_lr, save_dir=save_dir,
        model_name=model_name, class_weight=class_weight
    )
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, device, mode, num_classes)
    
    # Save model
    model_save_path = os.path.join(save_dir, f"{model_name}_{mode}_final_model.pth")
    
    metadata = {
        'model_name': model_name,
        'mode': mode,
        'in_channels': in_channels,
        'num_classes': num_classes,
        'task_type': f'{mode}_segmentation',
        'input_size': (patch_size, patch_size),
        'normalization': 'percentile_99',
        'encoder_name': encoder_name,
        'pretrained': pretrained,
        'dropout_rate': dropout_rate,
        'data_augmentation': data_augmentation,
        'use_class_weights': use_class_weights,
        'training_params': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'adjusted_learning_rate': adjusted_lr,
            'best_val_loss': best_loss,
            'best_val_iou': best_iou,
        },
        'performance_metrics': test_metrics,
        'complexity_info': complexity_info
    }
    
    save_model(model, model_save_path, metadata)
    
    # Save detailed metrics
    metrics = {
        'model_name': model_name,
        'mode': mode,
        'encoder_name': encoder_name,
        'in_channels': in_channels,
        'num_classes': num_classes,
        'patch_size': patch_size,
        'data_augmentation': data_augmentation,
        'use_class_weights': use_class_weights,
        'complexity_info': complexity_info,
        'test_metrics': test_metrics,
        'best_val_loss': best_loss,
        'best_val_iou': best_iou,
        'training_history': history
    }
    
    metrics_save_path = os.path.join(save_dir, f"{model_name}_{mode}_metrics.json")
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot training history
    plot_save_path = os.path.join(save_dir, f"{model_name}_{mode}_training_plot.png")
    plot_training_history(history, plot_save_path, mode)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED ({mode.upper()})")
    print(f"{'='*60}")
    
    if mode == 'binary':
        print(f"  Best validation loss: {best_loss:.4f}")
        print(f"  Best validation IoU: {best_iou:.4f}")
        print(f"  Test IoU: {test_metrics['iou']:.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    else:
        print(f"  Best validation loss: {best_loss:.4f}")
        print(f"  Best validation mIoU: {best_iou:.4f}")
        print(f"  Test mIoU (macro): {test_metrics.get('iou_macro', test_metrics.get('mean_iou', 0)):.4f}")
        print(f"  Test F1 (macro): {test_metrics.get('f1_macro', 0):.4f}")
        print(f"  Test Accuracy (macro): {test_metrics.get('accuracy_macro', 0):.4f}")
    
    print(f"\nSaved models:")
    print(f"  - Final: {model_save_path}")
    print(f"  - Best loss: {save_dir}/{model_name}_best_loss.pth")
    print(f"  - Best IoU: {save_dir}/{model_name}_best_iou.pth")
    print(f"  - Best combined: {save_dir}/{model_name}_best_combined.pth")
    
    return metrics

# =============== MODEL COMPATIBILITY FOR PREDICT_LARGE_IMAGE ===============
def build_model_for_prediction(name: str, in_channels: int = 4, classes: int = 6, 
                              encoder_name: str = 'resnet34', pretrained: bool = False,
                              dropout_rate: float = 0.5):
    """
    Build model specifically for compatibility with predict_large_image.py
    This function maintains the same signature as the old build_model
    """
    # Auto-detect mode based on classes
    mode = 'binary' if classes == 1 else 'multiclass'
    
    return ModelFactory.build_model(
        name=name,
        in_channels=in_channels,
        classes=classes,
        encoder_name=encoder_name,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        mode=mode
    )

# =============== AVAILABLE MODELS ===============
def get_available_models() -> List[str]:
    """Get list of available models"""
    models = ['unet-dropout']
    
    if HAS_SMP:
        models.extend([
            'unet', 'unet++', 'fpn', 'pspnet', 'linknet', 
            'manet', 'pan', 'deeplabv3', 'deeplabv3+'
        ])
    
    if HAS_TORCHVISION:
        models.append('deeplabv3')
    
    return models

if __name__ == "__main__":
    # Example usage
    models = get_available_models()
    print(f"Available models: {models}")
    
    config = SegmentationConfig.get_default_config('binary')
    print(f"Default binary config: {config}")
    
    config = SegmentationConfig.get_default_config('multiclass', num_classes=6)
    print(f"Default multiclass config: {config}")