"""
Advanced model training with .tif dataset loader - BINARY VERSION
-----------------------------------------------------------
Features:
- Loads .tif image datasets for train/val/test
- Trains single segmentation models with advanced features for BINARY segmentation
- Implements dropout, checkpointing, early stopping, LR scheduling
- Uses robust evaluation metrics
- Saves trained models with comprehensive metrics and plots
"""

import os
import time
import json
import random
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff

# Optional libraries
try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

try:
    from torchvision.models.segmentation import deeplabv3_resnet50
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# --------------- Data Augmentation for Multi-Channel Satellite Imagery ---------------
class MultiChannelAugmentation:
    """Advanced data augmentation for multi-channel satellite imagery with fine control"""
    
    def __init__(self, 
                 n_channels=10,
                 augmentation_prob=0.8,
                 geometric_aug=True,
                 noise_aug=True,
                 brightness_aug=True,
                 channel_group_aug=True):
        
        self.n_channels = n_channels
        self.augmentation_prob = augmentation_prob
        self.geometric_aug = geometric_aug
        self.noise_aug = noise_aug
        self.brightness_aug = brightness_aug
        self.channel_group_aug = channel_group_aug
        
        # Define channel groups for Sentinel-2
        self.channel_groups = {
            'coastal': [0],      # B1 - Coastal aerosol
            'blue': [1],         # B2 - Blue
            'green': [2],        # B3 - Green  
            'red': [3],          # B4 - Red
            'red_edge': [4, 5, 6],  # B5, B6, B7 - Red Edge
            'nir': [7, 8],       # B8, B8A - NIR
            'swir': [9]          # B11 - SWIR
        }
        
    def __call__(self, img, mask):
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
            if self.channel_group_aug:
                transforms.append('channel_group')
                
            random.shuffle(transforms)
            
            for transform in transforms:
                if transform == 'flip_h' and random.random() < 0.5:
                    img = torch.flip(img, dims=[2])
                    mask = torch.flip(mask, dims=[2])
                    
                elif transform == 'flip_v' and random.random() < 0.5:
                    img = torch.flip(img, dims=[1])
                    mask = torch.flip(mask, dims=[1])
                    
                elif transform == 'rotate' and random.random() < 0.5:
                    angle = random.choice([0, 90, 180, 270])
                    img = self.rotate_discrete(img, angle)
                    mask = self.rotate_discrete(mask, angle)
                    
                elif transform == 'noise' and random.random() < 0.3:
                    # Correlated noise between channels (more realistic)
                    base_noise = torch.randn(1, img.shape[1], img.shape[2]) * 0.03
                    noise = base_noise.repeat(img.shape[0], 1, 1)
                    img = img + noise
                    
                elif transform == 'brightness' and random.random() < 0.4:
                    # Consistent variation across all channels
                    factor = random.uniform(0.85, 1.15)
                    img = img * factor
                    
                elif transform == 'channel_group' and random.random() < 0.3:
                    # Variation by spectral group
                    group_name = random.choice(list(self.channel_groups.keys()))
                    channel_indices = self.channel_groups[group_name]
                    factor = random.uniform(0.9, 1.1)
                    
                    for idx in channel_indices:
                        if idx < img.shape[0]:
                            img[idx] = img[idx] * factor
            
            # Final clipping
            img = torch.clamp(img, 0, 1)
            
        except Exception as e:
            # In case of error, return original image
            print(f"Data augmentation error: {e}")
            return original_img, mask
        
        return img, mask
    
    def rotate_discrete(self, tensor, angle):
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

# --------------- Utils ---------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, device, input_size=(1, 10, 224, 224), n_runs=50):
    """Measure inference time with configurable input size"""
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Precise measurement
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
    print(f"Inference time: {avg_time:.4f}s")
    return avg_time

def get_model_complexity(model, device, input_size=(1, 10, 224, 224)):
    """Get model complexity metrics with configurable input size"""
    print("\nModel complexity analysis:")
    
    params = count_parameters(model)
    print(f"  - Parameters: {params:,}")
    
    inf_time = measure_inference_time(model, device, input_size=input_size)
    
    return {
        'parameters': params,
        'inference_time': inf_time
    }

def save_model(model, model_path, metadata=None):
    """Save model in standardized format"""
    model_info = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'model_architecture': str(model),
        'metadata': metadata or {}
    }
    torch.save(model_info, model_path)
    print(f"Model saved: {model_path}")

# --------------- Dataset - BINARY VERSION ---------------
class BinarySegmentationDataset(Dataset):
    def __init__(self, folder, image_subdir='images', mask_subdir='labels', transform=None):
        self.image_dir = Path(folder) / image_subdir
        self.mask_dir = Path(folder) / mask_subdir
        self.transform = transform

        print(f"Searching in:")
        print(f"  - Image dir: {self.image_dir}")
        print(f"  - Mask dir: {self.mask_dir}")

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
            print("Warning: Different number of images and masks!")
            print(f"  Images: {len(self.images)}, Masks: {len(self.masks)}")

    def __len__(self):
        return min(len(self.images), len(self.masks))

    def __getitem__(self, idx):
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
            mask = tiff.imread(mask_path).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Error reading file {img_path} or {mask_path}: {e}")

        # Verify mask is binary (0 and 1)
        unique_vals = np.unique(mask)
        if not set(unique_vals).issubset({0, 1}):
            print(f"Warning: mask {mask_path.name} contains non-binary values: {unique_vals}")
            # Force binarization: anything not 0 becomes 1
            mask = (mask > 0).astype(np.float32)

        # Image normalization - robust for Sentinel-2 data
        if np.max(img) > 0:
            # Normalization by percentile to reduce impact of outliers
            p99 = np.percentile(img, 99)
            img = np.clip(img / p99, 0, 1)
        else:
            img = np.zeros_like(img)

        # Handle channels
        if img.ndim == 2:
            img = img[..., None]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

# --------------- U-Net with Dropout - BINARY VERSION ---------------
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

class UNetBinary(nn.Module):
    def __init__(self, in_channels=10, out_channels=1, dropout_rate=0.5):
        super().__init__()
        
        # Encoder with reduced complexity to avoid overfitting
        self.enc1 = DoubleConv(in_channels, 32, dropout_rate)
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.enc2 = DoubleConv(32, 64, dropout_rate)
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.enc3 = DoubleConv(64, 128, dropout_rate)
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Center with reduced complexity
        self.center = DoubleConv(128, 256, dropout_rate)
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec3 = DoubleConv(256 + 128, 128, dropout_rate)
        self.dec2 = DoubleConv(128 + 64, 64, dropout_rate)
        self.dec1 = DoubleConv(64 + 32, 32, dropout_rate)

        # Remove final sigmoid - loss handles logits
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

        # Return logits (no sigmoid)
        return self.final(d1)

# --------------- Training Callbacks ---------------
class TrainingCallbacks:
    def __init__(self, save_dir, model_name, patience=15, min_delta=1e-3):
        self.save_dir = save_dir
        self.model_name = model_name
        self.patience = patience
        self.min_delta = min_delta
        
        self.best_loss = float('inf')
        self.best_iou = 0.0
        self.best_f1 = 0.0
        self.counter = 0
        self.early_stop = False
        
    def on_epoch_end(self, epoch, model, val_loss, val_iou, val_f1, optimizer, scheduler):
        improvement_found = False
        
        # Check improvement on multiple metrics
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
            }, checkpoint_path)
            print(f"Saved best_loss checkpoint")
            
        # Save best IoU model
        if val_iou > self.best_iou + self.min_delta:
            print(f"New best IoU: {val_iou:.4f} (previous: {self.best_iou:.4f})")
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
                }
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

# --------------- Loss Functions - BINARY VERSION ---------------
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-6, pos_weight=None):
        super().__init__()
        self.smooth = smooth
        # Use BCEWithLogitsLoss with pos_weight to handle imbalance
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        # Dice loss - use sigmoid for probabilities
        pred_sigmoid = torch.sigmoid(pred)
        
        intersection = (pred_sigmoid * target).sum(dim=(1, 2, 3))
        union = pred_sigmoid.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return bce_loss + dice_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = focal_weight * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()

# --------------- Model Factory - BINARY VERSION ---------------
def build_model(name, in_channels=10, classes=1, encoder_name='resnet34', pretrained=False, dropout_rate=0.5):
    name = name.lower()
    
    # U-Net with dropout - BINARY
    if name == 'unet-dropout':
        return UNetBinary(in_channels, classes, dropout_rate=dropout_rate)

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
            return smp_map[name](
                encoder_name=encoder_name, 
                in_channels=in_channels, 
                classes=classes,
                encoder_weights='imagenet' if pretrained else None,
                activation=None  # No activation for use with BCEWithLogitsLoss
            )

    if name == 'deeplabv3' and HAS_TORCHVISION:
        model = deeplabv3_resnet50(pretrained=False, num_classes=classes)
        if in_channels != 3:
            conv1 = model.backbone.conv1
            new_conv = nn.Conv2d(in_channels, conv1.out_channels, kernel_size=conv1.kernel_size, 
                               stride=conv1.stride, padding=conv1.padding, bias=conv1.bias is not None)
            with torch.no_grad():
                for i in range(min(in_channels, 3)):
                    new_conv.weight[:, i, :, :] = conv1.weight[:, i, :, :]
                if in_channels > 3:
                    for j in range(3, in_channels):
                        new_conv.weight[:, j, :, :] = conv1.weight[:, 0, :, :]
            model.backbone.conv1 = new_conv
        
        # Remove final sigmoid
        return model

    raise ValueError(f"Model {name} not supported. Install segmentation_models_pytorch or use 'unet-dropout'.")

# --------------- Metrics - BINARY VERSION ---------------
def compute_basic_binary_metrics(output, target, threshold=0.5):
    """Fallback metrics for binary segmentation"""
    # Apply sigmoid if not already done (for models returning logits)
    pred_probs = torch.sigmoid(output)
    pred = (pred_probs > threshold).float()
    
    # Calculate basic binary metrics
    tp = ((pred == 1) & (target == 1)).sum().float()
    tn = ((pred == 0) & (target == 0)).sum().float()
    fp = ((pred == 1) & (target == 0)).sum().float()
    fn = ((pred == 0) & (target == 1)).sum().float()
    
    # Avoid division by zero
    epsilon = 1e-6
    
    # IoU (Intersection over Union)
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

def compute_binary_metrics(output, target, threshold=0.5):
    """Compute metrics for binary segmentation with robust error handling"""
    try:
        # Handle different output shapes
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
        
        # Calculate stats with binary mode
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred, 
            target, 
            mode='binary',
            threshold=threshold
        )
        
        # Calculate metrics
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
        # Fallback to basic metrics
        return compute_basic_binary_metrics(output, target, threshold)

# --------------- Evaluation - BINARY VERSION ---------------
def evaluate_binary_model(model, dataloader, device, threshold=0.5):
    """Standard evaluation with metrics for binary"""
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            
            if isinstance(out, dict):
                out = out['out']
                
            all_outputs.append(out.cpu())
            all_targets.append(masks.cpu())
    
    # Concatenate all batches
    outputs = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = compute_binary_metrics(outputs, targets, threshold)
    return metrics

def evaluate_binary_model_with_loss(model, dataloader, device, criterion, threshold=0.5):
    """Evaluation with loss and metrics for binary"""
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
    
    # Calculate average loss
    mean_loss = total_loss / len(dataloader.dataset)
    
    # Concatenate and calculate metrics
    outputs = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_binary_metrics(outputs, targets, threshold)
    
    return {
        'loss': mean_loss,
        'iou': metrics['iou'],
        'f1': metrics['f1'],
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'detailed_metrics': metrics
    }

# --------------- Calculate Class Weights ---------------
def calculate_class_weights(dataloader, device):
    """Calculate class weights to handle imbalance"""
    total_pixels = 0
    positive_pixels = 0
    
    for _, masks in dataloader:
        positive_pixels += (masks > 0).sum().item()
        total_pixels += masks.numel()
    
    negative_pixels = total_pixels - positive_pixels
    positive_weight = negative_pixels / (positive_pixels + 1e-6)  # Avoid division by zero
    
    print(f"Class statistics:")
    print(f"  - Positive pixels: {positive_pixels} ({positive_pixels/total_pixels*100:.2f}%)")
    print(f"  - Negative pixels: {negative_pixels} ({negative_pixels/total_pixels*100:.2f}%)")
    print(f"  - Positive class weight: {positive_weight:.2f}")
    
    return torch.tensor([positive_weight], device=device)

# --------------- Training - BINARY VERSION ---------------
def train_binary_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3, 
                       save_dir='./saved_models', model_name='model', class_weight=None):
    model.to(device)
    
    # Optimizer with reduced weight decay for small dataset
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Loss with imbalance handling
    if class_weight is not None:
        criterion = DiceBCELoss(pos_weight=class_weight)
    else:
        criterion = DiceBCELoss()
    
    # More aggressive scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    
    # Callbacks with increased patience and multi-metric monitoring
    callbacks = TrainingCallbacks(save_dir, model_name, patience=15, min_delta=1e-3)
    
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': [], 
               'val_accuracy': [], 'val_precision': [], 'val_recall': [], 
               'epoch_time': [], 'learning_rate': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, start_time = 0, time.perf_counter()
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            
            # More permissive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        val_metrics = evaluate_binary_model_with_loss(model, val_loader, device, criterion)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['epoch_time'].append(time.perf_counter() - start_time)
        history['learning_rate'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Loss: train={train_loss:.4f}, val={val_metrics['loss']:.4f}")
        print(f"  IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        print(f"  LR: {current_lr:.2e}, Time: {history['epoch_time'][-1]:.1f}s")
        
        # Callbacks with multi-metric monitoring
        early_stop = callbacks.on_epoch_end(
            epoch, model, val_metrics['loss'], val_metrics['iou'], val_metrics['f1'], optimizer, scheduler
        )
        
        if early_stop:
            print("Early stopping training")
            break
    
    return model, history, callbacks.best_loss, callbacks.best_iou

# --------------- Visualization - BINARY VERSION ---------------
def plot_binary_training_history(history, save_path):
    """Plot training history for binary segmentation"""
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
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# --------------- Main Training Function - BINARY VERSION ---------------
def train_and_save_binary_model(model_name, dataset_root, device='cuda', batch_size=4, epochs=100, 
                                save_dir='./saved_models', encoder_name='resnet34', pretrained=False,
                                dropout_rate=0.5, learning_rate=1e-3, in_channels=10, patch_size=224,
                                data_augmentation=False, use_class_weights=True):
    """Train a single model with advanced features for BINARY segmentation"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Multi-channel data augmentation
    if data_augmentation:
        if in_channels > 3:
            # Use advanced version for multi-channels
            augmentation = MultiChannelAugmentation(
                n_channels=in_channels,
                augmentation_prob=0.7,
                geometric_aug=True,
                noise_aug=True,
                brightness_aug=True,
                channel_group_aug=True
            )
            print(f"Multi-channel data augmentation enabled ({in_channels} channels)")
        else:
            # Simple version for RGB
            augmentation = MultiChannelAugmentation(
                n_channels=in_channels,
                augmentation_prob=0.7
            )
            print("Standard data augmentation enabled")
    else:
        augmentation = None
        print("Data augmentation disabled")
    
    # Load datasets
    train_dataset = BinarySegmentationDataset(os.path.join(dataset_root, 'Patch', 'train'), transform=augmentation)
    val_dataset = BinarySegmentationDataset(os.path.join(dataset_root, 'Patch', 'validation'))
    test_dataset = BinarySegmentationDataset(os.path.join(dataset_root, 'Patch', 'test'))

    # Adjust batch_size for small dataset
    effective_batch_size = min(batch_size, len(train_dataset))
    if effective_batch_size != batch_size:
        print(f"Batch_size adjusted from {batch_size} to {effective_batch_size} (dataset size)")
    
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Calculate class weights
    class_weight = None
    if use_class_weights:
        class_weight = calculate_class_weights(train_loader, device)
    
    # Build model with reduced complexity
    print(f"\n=== Training {model_name} (Multi-Channel - {in_channels} channels) ===")
    model = build_model(model_name, in_channels=in_channels, classes=1,
                       encoder_name=encoder_name, pretrained=pretrained,
                       dropout_rate=dropout_rate)
    
    # Adjust learning rate for small dataset
    adjusted_lr = learning_rate * 0.5
    
    print(f"\nOptimized multi-channel configuration:")
    print(f"  - Input channels: {in_channels}")
    print(f"  - Dataset size: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"  - Effective batch size: {effective_batch_size}")
    print(f"  - Adjusted learning rate: {adjusted_lr:.1e}")
    print(f"  - Data augmentation: {'Multi-channel' if data_augmentation else 'Disabled'}")
    print(f"  - Class weights: {'Enabled' if use_class_weights else 'Disabled'}")
    if class_weight is not None:
        print(f"  - Positive class weight: {class_weight.item():.2f}")
    
    # Compute model stats
    input_size = (1, in_channels, patch_size, patch_size)
    complexity_info = get_model_complexity(model, device, input_size=input_size)
    
    # Train model with advanced features for binary
    model, history, best_loss, best_iou = train_binary_model(
        model, train_loader, val_loader, device, epochs=epochs, 
        lr=adjusted_lr, save_dir=save_dir, model_name=model_name,
        class_weight=class_weight
    )
    
    # Evaluate on test set with metrics for binary
    test_metrics = evaluate_binary_model(model, test_loader, device)
    
    # Save model
    model_save_path = os.path.join(save_dir, f"{model_name}_final_model.pth")
    
    metadata = {
        'model_name': model_name,
        'in_channels': in_channels,
        'num_classes': 1,
        'task_type': 'binary_segmentation',
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
        'performance_metrics': {
            'test_iou': test_metrics['iou'],
            'test_f1': test_metrics['f1'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'parameters': complexity_info['parameters'],
            'inference_time': complexity_info['inference_time'],
        }
    }
    
    save_model(model, model_save_path, metadata)
    
    # Save detailed metrics
    metrics = {
        'model_name': model_name,
        'encoder_name': encoder_name,
        'in_channels': in_channels,
        'task_type': 'binary_segmentation',
        'patch_size': patch_size,
        'data_augmentation': data_augmentation,
        'use_class_weights': use_class_weights,
        'complexity_info': complexity_info,
        'test_metrics': test_metrics,
        'best_val_loss': best_loss,
        'best_val_iou': best_iou,
        'training_history': history
    }
    
    metrics_save_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot and save training history for binary
    plot_save_path = os.path.join(save_dir, f"{model_name}_training_plot.png")
    plot_binary_training_history(history, plot_save_path)
    
    print(f"\nTraining completed with advanced features (BINARY)!")
    print(f"  Best validation loss: {best_loss:.4f}")
    print(f"  Best validation IoU: {best_iou:.4f}")
    print(f"  Test IoU: {test_metrics['iou']:.4f}")
    print(f"  Test F1: {test_metrics['f1']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall: {test_metrics['recall']:.4f}")
    print(f"  Data augmentation: {'Multi-channel' if data_augmentation else 'Disabled'}")
    print(f"  Class weights: {'Enabled' if use_class_weights else 'Disabled'}")
    print(f"  Saved models:")
    print(f"    - Final: {model_save_path}")
    print(f"    - Best loss: {save_dir}/{model_name}_best_loss.pth}")
    print(f"    - Best IoU: {save_dir}/{model_name}_best_iou.pth")
    print(f"    - Best combined: {save_dir}/{model_name}_best_combined.pth")
    
    return metrics

# --------------- Available Models ---------------
def get_available_models():
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
    models = get_available_models()
    print(f"Available models: {models}")