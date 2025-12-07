"""
Advanced model training with .tif dataset loader
-----------------------------------------------------------
Features:
- Loads .tif image datasets for train/val/test
- Trains segmentation models with advanced features
- Implements dropout, checkpointing, early stopping, LR scheduling
- Uses robust evaluation metrics
- Saves trained models with comprehensive metrics and plots
"""

import os
import time
import json
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

# --------------- Utils ---------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, device, input_size=(1, 4, 224, 224), n_runs=50):
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
    print(f"Inference time: {avg_time:.4f}s")
    return avg_time

def get_model_complexity(model, device):
    """Get model complexity metrics"""
    print("\nModel complexity analysis:")
    
    params = count_parameters(model)
    print(f"  - Parameters: {params:,}")
    
    inf_time = measure_inference_time(model, device)
    
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

# --------------- Dataset ---------------
class SegmentationDataset(Dataset):
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
            mask = tiff.imread(mask_path).astype(np.int64)
        except Exception as e:
            raise RuntimeError(f"Error reading file {img_path} or {mask_path}: {e}")

        # Normalize
        if np.max(img) > 0:
            img = img / np.max(img)
        else:
            img = np.zeros_like(img)

        # Handle channels
        if img.ndim == 2:
            img = img[..., None]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

# --------------- U-Net with Dropout ---------------
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

class UNetWithDropout(nn.Module):
    def __init__(self, in_channels=4, out_channels=6, dropout_rate=0.5):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64, dropout_rate)
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.enc2 = DoubleConv(64, 128, dropout_rate)
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.enc3 = DoubleConv(128, 256, dropout_rate)
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.enc4 = DoubleConv(256, 512, dropout_rate)
        self.pool4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Center
        self.center = DoubleConv(512, 1024, dropout_rate)
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec4 = DoubleConv(1024 + 512, 512, dropout_rate)
        self.dec3 = DoubleConv(512 + 256, 256, dropout_rate)
        self.dec2 = DoubleConv(256 + 128, 128, dropout_rate)
        self.dec1 = DoubleConv(128 + 64, 64, dropout_rate)

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

# --------------- Training Callbacks ---------------
class TrainingCallbacks:
    def __init__(self, save_dir, model_name, patience=15, min_delta=1e-4):
        self.save_dir = save_dir
        self.model_name = model_name
        self.patience = patience
        self.min_delta = min_delta
        
        self.best_loss = float('inf')
        self.best_miou = 0.0
        self.best_f1 = 0.0
        self.counter = 0
        self.early_stop = False
        
    def on_epoch_end(self, epoch, model, val_loss, val_miou, val_f1, optimizer, scheduler):
        improvement_found = False
        
        loss_improved = False
        miou_improved = False
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
                'val_miou': val_miou,
                'val_f1': val_f1,
                'best_loss': self.best_loss,
            }, checkpoint_path)
            print(f"Saved best_loss checkpoint")
            
        # Save best mIoU model
        if val_miou > self.best_miou + self.min_delta:
            print(f"New best mIoU: {val_miou:.4f} (previous: {self.best_miou:.4f})")
            self.best_miou = val_miou
            improvement_found = True
            miou_improved = True
            
            checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_best_miou.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_miou': val_miou,
                'val_f1': val_f1,
                'best_miou': self.best_miou,
            }, checkpoint_path)
            print(f"Saved best_miou checkpoint")
        
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
                'val_miou': val_miou,
                'val_f1': val_f1,
                'best_loss': self.best_loss,
                'best_miou': self.best_miou,
                'best_f1': self.best_f1,
                'improvements': {
                    'loss_improved': loss_improved,
                    'miou_improved': miou_improved,
                    'f1_improved': f1_improved
                }
            }, checkpoint_path)
            print(f"Saved combined checkpoint (improvements: loss={loss_improved}, miou={miou_improved}, f1={f1_improved})")
            
        else:
            # No improvement on main metrics
            self.counter += 1
            print(f"No significant improvement - Patience: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered after {epoch+1} epochs")
                
        return self.early_stop

# --------------- Loss Function ---------------
class DiceCELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        
        # Dice loss
        pred_softmax = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()
        
        return ce_loss + dice_loss

# --------------- Model Factory ---------------
def build_model(name, in_channels=4, classes=6, encoder_name='resnet34', pretrained=False, dropout_rate=0.5):
    name = name.lower()
    
    # Custom U-Net with dropout
    if name == 'unet-dropout':
        return UNetWithDropout(in_channels, classes, dropout_rate=dropout_rate)

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
                encoder_weights='imagenet' if pretrained else None
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
        return model

    raise ValueError(f"Model {name} not supported. Install segmentation_models_pytorch or use 'unet-dropout'.")

# --------------- Metrics ---------------
def compute_basic_metrics(output, target, num_classes):
    """Fallback metrics if SMP fails"""
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

def compute_metrics(output, target, num_classes):
    """Compute metrics using SMP with error handling"""
    try:
        if output.dim() == 4 and target.dim() == 3:
            if not torch.is_tensor(output):
                output = torch.from_numpy(output)
            if not torch.is_tensor(target):
                target = torch.from_numpy(target)
                
            pred = output.argmax(dim=1)
            target = target.long()
            
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred,
                target, 
                mode='multiclass',
                num_classes=num_classes
            )
            
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
        return compute_basic_metrics(output, target, num_classes)

# --------------- Evaluation ---------------
def evaluate_model(model, dataloader, device, num_classes):
    """Standard evaluation with metrics"""
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
    
    outputs = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(outputs, targets, num_classes)
    return metrics

def evaluate_model_with_loss(model, dataloader, device, criterion, num_classes):
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
    metrics = compute_metrics(outputs, targets, num_classes)
    
    return {
        'loss': mean_loss,
        'mean_iou': metrics['iou_macro'],
        'mean_f1': metrics['f1_macro'],
        'accuracy': metrics['accuracy_macro'],
        'detailed_metrics': metrics
    }

# --------------- Training ---------------
def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3, 
                save_dir='./saved_models', model_name='model'):
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = DiceCELoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    callbacks = TrainingCallbacks(save_dir, model_name, patience=15, min_delta=1e-4)
    
    history = {'train_loss': [], 'val_loss': [], 'val_miou': [], 'val_f1': [], 
               'val_accuracy': [], 'epoch_time': [], 'learning_rate': []}
    
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
        val_metrics = evaluate_model_with_loss(model, val_loader, device, criterion, num_classes=6)
        
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_miou'].append(val_metrics['mean_iou'])
        history['val_f1'].append(val_metrics['mean_f1'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['epoch_time'].append(time.perf_counter() - start_time)
        history['learning_rate'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Loss: train={train_loss:.4f}, val={val_metrics['loss']:.4f}")
        print(f"  mIoU: {val_metrics['mean_iou']:.4f}, F1: {val_metrics['mean_f1']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  LR: {current_lr:.2e}, Time: {history['epoch_time'][-1]:.1f}s")
        
        early_stop = callbacks.on_epoch_end(
            epoch, model, val_metrics['loss'], val_metrics['mean_iou'], val_metrics['mean_f1'], optimizer, scheduler
        )
        
        if early_stop:
            print("Early stopping")
            break
    
    return model, history, callbacks.best_loss, callbacks.best_miou

# --------------- Visualization ---------------
def plot_training_history(history, save_path):
    """Plot training history"""
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
    ax2.plot(history['val_miou'], 'g-', label='Validation mIoU')
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

# --------------- Main Training Function ---------------
def train_and_save_model(model_name, dataset_root, device='cuda', batch_size=4, epochs=100, 
                         save_dir='./saved_models', encoder_name='resnet34', pretrained=False,
                         dropout_rate=0.5, learning_rate=1e-3):
    """Train a single model with advanced features"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load datasets
    train_dataset = SegmentationDataset(os.path.join(dataset_root, 'Patch', 'train'))
    val_dataset = SegmentationDataset(os.path.join(dataset_root, 'Patch', 'validation'))
    test_dataset = SegmentationDataset(os.path.join(dataset_root, 'Patch', 'test'))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)

    print(f"\n=== Training {model_name} ===")
    model = build_model(model_name, in_channels=4, classes=6, 
                        encoder_name=encoder_name, pretrained=pretrained,
                        dropout_rate=dropout_rate)
    
    complexity_info = get_model_complexity(model, device)
    
    print(f"\nModel complexity:")
    print(f"  - Parameters: {complexity_info['parameters']:,}")
    print(f"  - Inference time: {complexity_info['inference_time']:.4f}s")
    print(f"  - Dropout rate: {dropout_rate}")
    print(f"  - Learning rate: {learning_rate}")
    
    # Train model
    model, history, best_loss, best_miou = train_model(
        model, train_loader, val_loader, device, epochs=epochs, 
        lr=learning_rate, save_dir=save_dir, model_name=model_name
    )
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, device, num_classes=6)
    
    # Save model
    model_save_path = os.path.join(save_dir, f"{model_name}_final_model.pth")
    
    metadata = {
        'model_name': model_name,
        'in_channels': 4,
        'num_classes': 6, 
        'encoder_name': encoder_name,
        'pretrained': pretrained,
        'dropout_rate': dropout_rate,
        'training_params': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_val_loss': best_loss,
            'best_val_miou': best_miou,
        },
        'performance_metrics': {
            'test_miou_micro': test_metrics['iou_micro'],
            'test_miou_macro': test_metrics['iou_macro'],
            'test_f1_micro': test_metrics['f1_micro'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_accuracy_micro': test_metrics['accuracy_micro'],
            'test_accuracy_macro': test_metrics['accuracy_macro'],
            'parameters': complexity_info['parameters'],
            'inference_time': complexity_info['inference_time'],
        }
    }
    
    save_model(model, model_save_path, metadata)
    
    # Save detailed metrics
    metrics = {
        'model_name': model_name,
        'encoder_name': encoder_name,
        'complexity_info': complexity_info,
        'test_metrics': test_metrics,
        'best_val_loss': best_loss,
        'best_val_miou': best_miou,
        'training_history': history
    }
    
    metrics_save_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot and save training history
    plot_save_path = os.path.join(save_dir, f"{model_name}_training_plot.png")
    plot_training_history(history, plot_save_path)
    
    print(f"\nTraining completed!")
    print(f"  Best validation loss: {best_loss:.4f}")
    print(f"  Best validation mIoU: {best_miou:.4f}")
    print(f"  Test mIoU (macro): {test_metrics['iou_macro']:.4f}")
    print(f"  Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"  Test Accuracy (macro): {test_metrics['accuracy_macro']:.4f}")
    print(f"  Saved models:")
    print(f"    - Final: {model_save_path}")
    print(f"    - Best loss: {save_dir}/{model_name}_best_loss.pth")
    print(f"    - Best mIoU: {save_dir}/{model_name}_best_miou.pth")
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