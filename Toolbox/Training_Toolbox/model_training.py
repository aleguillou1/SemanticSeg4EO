"""
UNIFIED ADVANCED SEGMENTATION TRAINING SYSTEM V2
-----------------------------------------------------------
Enhanced Features:
- Per-class IoU logging and visualization
- Multiple loss functions: Focal, Dice, Combo, Tversky
- Encoder freezing for transfer learning
- Learning rate warmup
- Mixed precision training (AMP)
- CSV metrics export
- Improved early stopping with per-class monitoring

Author: Enhanced version
"""

import os
import time
import json
import csv
import random
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
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


# =============== CONFIGURATION DATACLASS ===============
@dataclass
class TrainingConfig:
    """Complete training configuration with all options"""
    
    # Core settings
    mode: str = 'multiclass'
    num_classes: int = 5
    in_channels: int = 3
    patch_size: int = 224
    
    # Model settings
    model_name: str = 'unet++'
    encoder_name: str = 'resnet34'
    pretrained: bool = True
    dropout_rate: float = 0.3
    
    # Training settings
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    
    # Loss settings
    loss_type: str = 'dice_ce'  # Options: 'ce', 'dice', 'dice_ce', 'focal', 'focal_dice', 'tversky', 'combo'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3
    dice_weight: float = 0.5  # Weight for dice in combo losses
    ce_weight: float = 0.5    # Weight for CE in combo losses
    
    # Class weights
    use_class_weights: bool = True
    class_weight_method: str = 'inverse_freq'  # Options: 'inverse_freq', 'effective_samples', 'custom'
    custom_class_weights: Optional[List[float]] = None
    
    # Encoder freezing
    freeze_encoder: bool = False
    freeze_epochs: int = 5  # Number of epochs to keep encoder frozen
    
    # Learning rate schedule
    scheduler_type: str = 'reduce_plateau'  # Options: 'reduce_plateau', 'cosine', 'one_cycle'
    warmup_epochs: int = 0  # Number of warmup epochs
    warmup_lr: float = 1e-6  # Initial LR for warmup
    
    # Mixed precision
    use_amp: bool = False  # Automatic Mixed Precision
    
    # Data augmentation
    data_augmentation: bool = True
    augmentation_prob: float = 0.8
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    
    # Logging
    log_per_class_metrics: bool = True  # NEW: Log IoU per class
    save_csv_logs: bool = True  # NEW: Save metrics to CSV
    class_names: Optional[List[str]] = None  # Optional class names for logging
    
    # Paths
    save_dir: str = './trained_models'
    
    def __post_init__(self):
        """Validate configuration"""
        valid_losses = ['ce', 'dice', 'dice_ce', 'focal', 'focal_dice', 'tversky', 'combo']
        if self.loss_type not in valid_losses:
            raise ValueError(f"Invalid loss_type: {self.loss_type}. Must be one of {valid_losses}")
        
        valid_schedulers = ['reduce_plateau', 'cosine', 'one_cycle']
        if self.scheduler_type not in valid_schedulers:
            raise ValueError(f"Invalid scheduler_type: {self.scheduler_type}. Must be one of {valid_schedulers}")
        
        if self.mode == 'binary':
            self.num_classes = 1


# =============== LOSS FUNCTIONS ===============
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 weight: Optional[torch.Tensor] = None, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, weight=self.weight, 
                                   ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Pure Dice Loss"""
    
    def __init__(self, smooth: float = 1e-6, multiclass: bool = True):
        super().__init__()
        self.smooth = smooth
        self.multiclass = multiclass
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.multiclass:
            pred_softmax = F.softmax(pred, dim=1)
            num_classes = pred.shape[1]
            
            if target.dim() == 3:
                target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
            else:
                target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
            
            intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
            union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice.mean()
        else:
            pred_sigmoid = torch.sigmoid(pred)
            intersection = (pred_sigmoid * target).sum(dim=(1, 2, 3))
            union = pred_sigmoid.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice with alpha/beta control"""
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_softmax = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        
        if target.dim() == 3:
            target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        else:
            target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        tp = (pred_softmax * target_one_hot).sum(dim=(2, 3))
        fp = (pred_softmax * (1 - target_one_hot)).sum(dim=(2, 3))
        fn = ((1 - pred_softmax) * target_one_hot).sum(dim=(2, 3))
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()


class ComboLoss(nn.Module):
    """Flexible combination of multiple losses"""
    
    def __init__(self, losses: List[nn.Module], weights: List[float]):
        super().__init__()
        assert len(losses) == len(weights), "Number of losses must match number of weights"
        self.losses = nn.ModuleList(losses)
        self.weights = weights
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(pred, target)
        return total_loss


class LossFactory:
    """Factory for creating loss functions based on configuration"""
    
    @staticmethod
    def create_loss(config: TrainingConfig, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        """Create appropriate loss function based on config"""
        
        mode = config.mode
        loss_type = config.loss_type
        
        if mode == 'binary':
            return LossFactory._create_binary_loss(config, class_weights)
        else:
            return LossFactory._create_multiclass_loss(config, class_weights)
    
    @staticmethod
    def _create_binary_loss(config: TrainingConfig, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        """Create binary segmentation loss"""
        
        class DiceBCELoss(nn.Module):
            def __init__(self, pos_weight=None, dice_weight=0.5, smooth=1e-6):
                super().__init__()
                self.smooth = smooth
                self.dice_weight = dice_weight
                self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
            def forward(self, pred, target):
                bce_loss = self.bce(pred, target)
                pred_sigmoid = torch.sigmoid(pred)
                intersection = (pred_sigmoid * target).sum(dim=(1, 2, 3))
                union = pred_sigmoid.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
                dice = (2. * intersection + self.smooth) / (union + self.smooth)
                dice_loss = 1 - dice.mean()
                return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss
        
        return DiceBCELoss(pos_weight=class_weights, dice_weight=config.dice_weight)
    
    @staticmethod
    def _create_multiclass_loss(config: TrainingConfig, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        """Create multi-class segmentation loss"""
        
        loss_type = config.loss_type
        
        if loss_type == 'ce':
            return nn.CrossEntropyLoss(weight=class_weights)
        
        elif loss_type == 'dice':
            return DiceLoss(multiclass=True)
        
        elif loss_type == 'dice_ce':
            ce = nn.CrossEntropyLoss(weight=class_weights)
            dice = DiceLoss(multiclass=True)
            return ComboLoss([ce, dice], [config.ce_weight, config.dice_weight])
        
        elif loss_type == 'focal':
            return FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma, weight=class_weights)
        
        elif loss_type == 'focal_dice':
            focal = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma, weight=class_weights)
            dice = DiceLoss(multiclass=True)
            return ComboLoss([focal, dice], [config.ce_weight, config.dice_weight])
        
        elif loss_type == 'tversky':
            return TverskyLoss(alpha=config.tversky_alpha, beta=config.tversky_beta)
        
        elif loss_type == 'combo':
            # Full combo: CE + Dice + Focal
            ce = nn.CrossEntropyLoss(weight=class_weights)
            dice = DiceLoss(multiclass=True)
            focal = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma, weight=class_weights)
            return ComboLoss([ce, dice, focal], [0.4, 0.3, 0.3])
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


# =============== PER-CLASS METRICS ===============
class PerClassMetrics:
    """Compute and store per-class metrics"""
    
    @staticmethod
    def compute_per_class_iou(pred: torch.Tensor, target: torch.Tensor, 
                               num_classes: int) -> Dict[str, float]:
        """Compute IoU for each class"""
        
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        
        per_class_iou = {}
        iou_list = []
        
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            
            if union > 0:
                iou = (intersection / union).item()
            else:
                iou = float('nan')  # Class not present
            
            per_class_iou[f'iou_class_{cls}'] = iou
            if not np.isnan(iou):
                iou_list.append(iou)
        
        # Mean IoU (excluding NaN)
        per_class_iou['mean_iou'] = np.nanmean(iou_list) if iou_list else 0.0
        
        return per_class_iou
    
    @staticmethod
    def compute_per_class_f1(pred: torch.Tensor, target: torch.Tensor,
                              num_classes: int) -> Dict[str, float]:
        """Compute F1 for each class"""
        
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        
        per_class_f1 = {}
        f1_list = []
        
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            
            tp = (pred_cls & target_cls).sum().float()
            fp = (pred_cls & ~target_cls).sum().float()
            fn = (~pred_cls & target_cls).sum().float()
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            
            per_class_f1[f'f1_class_{cls}'] = f1.item()
            f1_list.append(f1.item())
        
        per_class_f1['mean_f1'] = np.mean(f1_list)
        
        return per_class_f1
    
    @staticmethod
    def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor,
                                  num_classes: int) -> np.ndarray:
        """Compute confusion matrix"""
        
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        
        pred_flat = pred.view(-1).cpu().numpy()
        target_flat = target.view(-1).cpu().numpy()
        
        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        for t, p in zip(target_flat, pred_flat):
            conf_matrix[t, p] += 1
        
        return conf_matrix


class MetricsFactory:
    """Factory for computing metrics - enhanced with per-class support"""
    
    @staticmethod
    def compute_metrics(mode: str, output: torch.Tensor, target: torch.Tensor, 
                        num_classes: int = 2, threshold: float = 0.5,
                        compute_per_class: bool = True) -> Dict:
        """Compute metrics with optional per-class breakdown"""
        
        if mode == 'binary':
            return MetricsFactory._compute_binary_metrics(output, target, threshold)
        else:
            return MetricsFactory._compute_multiclass_metrics(output, target, num_classes, compute_per_class)
    
    @staticmethod
    def _compute_binary_metrics(output: torch.Tensor, target: torch.Tensor, 
                                 threshold: float = 0.5) -> Dict:
        """Binary metrics"""
        try:
            if HAS_SMP:
                pred = (torch.sigmoid(output) > threshold).long()
                tp, fp, fn, tn = smp.metrics.get_stats(pred, target.long(), mode='binary', threshold=threshold)
                
                return {
                    'iou': smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item(),
                    'f1': smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item(),
                    'accuracy': smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item(),
                    'precision': smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item(),
                    'recall': smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item()
                }
        except Exception:
            pass
        
        # Fallback
        pred = (torch.sigmoid(output) > threshold).float()
        tp = ((pred == 1) & (target == 1)).sum().float()
        tn = ((pred == 0) & (target == 0)).sum().float()
        fp = ((pred == 1) & (target == 0)).sum().float()
        fn = ((pred == 0) & (target == 1)).sum().float()
        
        epsilon = 1e-6
        iou = tp / (tp + fp + fn + epsilon)
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        
        return {
            'iou': iou.item(),
            'f1': f1.item(),
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item()
        }
    
    @staticmethod
    def _compute_multiclass_metrics(output: torch.Tensor, target: torch.Tensor,
                                     num_classes: int, compute_per_class: bool = True) -> Dict:
        """Multi-class metrics with per-class breakdown"""
        
        metrics = {}
        
        try:
            if HAS_SMP and output.dim() == 4 and target.dim() == 3:
                pred = output.argmax(dim=1)
                tp, fp, fn, tn = smp.metrics.get_stats(pred, target.long(), mode='multiclass', num_classes=num_classes)
                
                metrics['iou_micro'] = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
                metrics['f1_micro'] = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
                metrics['accuracy_micro'] = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item()
                
                metrics['iou_macro'] = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro").item()
                metrics['f1_macro'] = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro").item()
                metrics['accuracy_macro'] = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").item()
                
                metrics['mean_iou'] = metrics['iou_macro']
        except Exception:
            # Fallback
            pred = output.argmax(dim=1) if output.dim() == 4 else output
            iou_list = []
            for cls in range(num_classes):
                pred_cls = (pred == cls)
                target_cls = (target == cls)
                intersection = (pred_cls & target_cls).sum().float()
                union = (pred_cls | target_cls).sum().float()
                if union > 0:
                    iou_list.append((intersection / union).item())
            
            metrics['mean_iou'] = np.mean(iou_list) if iou_list else 0.0
            metrics['iou_macro'] = metrics['mean_iou']
            metrics['f1_macro'] = metrics['mean_iou']
            metrics['accuracy_macro'] = (pred == target).sum().float().item() / target.numel()
            metrics['iou_micro'] = metrics['iou_macro']
            metrics['f1_micro'] = metrics['f1_macro']
            metrics['accuracy_micro'] = metrics['accuracy_macro']
        
        # Per-class metrics
        if compute_per_class:
            per_class_iou = PerClassMetrics.compute_per_class_iou(output, target, num_classes)
            per_class_f1 = PerClassMetrics.compute_per_class_f1(output, target, num_classes)
            metrics['per_class_iou'] = per_class_iou
            metrics['per_class_f1'] = per_class_f1
        
        return metrics


# =============== ENCODER FREEZING ===============
class EncoderFreezer:
    """Handle encoder freezing and unfreezing"""
    
    def __init__(self, model: nn.Module, freeze_epochs: int = 5):
        self.model = model
        self.freeze_epochs = freeze_epochs
        self.is_frozen = False
        self.encoder_params = []
        self._identify_encoder()
    
    def _identify_encoder(self):
        """Identify encoder parameters"""
        if hasattr(self.model, 'encoder'):
            # SMP models
            self.encoder_params = list(self.model.encoder.parameters())
        elif hasattr(self.model, 'backbone'):
            # TorchVision models
            self.encoder_params = list(self.model.backbone.parameters())
        else:
            # Try to find encoder-like modules
            for name, module in self.model.named_modules():
                if 'encoder' in name.lower() or 'backbone' in name.lower():
                    self.encoder_params.extend(list(module.parameters()))
                    break
    
    def freeze(self):
        """Freeze encoder parameters"""
        if not self.encoder_params:
            warnings.warn("No encoder parameters found to freeze")
            return
        
        for param in self.encoder_params:
            param.requires_grad = False
        self.is_frozen = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Encoder frozen: {trainable:,}/{total:,} parameters trainable ({100*trainable/total:.1f}%)")
    
    def unfreeze(self):
        """Unfreeze encoder parameters"""
        for param in self.encoder_params:
            param.requires_grad = True
        self.is_frozen = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Encoder unfrozen: {trainable:,}/{total:,} parameters trainable ({100*trainable/total:.1f}%)")
    
    def check_epoch(self, epoch: int):
        """Check if we should unfreeze at this epoch"""
        if self.is_frozen and epoch >= self.freeze_epochs:
            print(f"\n[Epoch {epoch+1}] Unfreezing encoder...")
            self.unfreeze()
            return True
        return False


# =============== LEARNING RATE WARMUP ===============
class WarmupScheduler:
    """Learning rate warmup wrapper"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int,
                 warmup_lr: float, target_lr: float, steps_per_epoch: int):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.target_lr = target_lr
        self.steps_per_epoch = steps_per_epoch
        self.total_warmup_steps = warmup_epochs * steps_per_epoch
        self.current_step = 0
    
    def step(self):
        """Update learning rate during warmup"""
        if self.current_step < self.total_warmup_steps:
            lr = self.warmup_lr + (self.target_lr - self.warmup_lr) * (self.current_step / self.total_warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_step += 1
    
    def is_warmup_done(self) -> bool:
        return self.current_step >= self.total_warmup_steps


# =============== CSV LOGGER ===============
class CSVLogger:
    """Log metrics to CSV file"""
    
    def __init__(self, save_dir: str, model_name: str, num_classes: int, class_names: Optional[List[str]] = None):
        self.save_dir = save_dir
        self.filepath = os.path.join(save_dir, f"{model_name}_training_log.csv")
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.initialized = False
    
    def _initialize(self, fieldnames: List[str]):
        """Initialize CSV file with headers"""
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        self.initialized = True
        self.fieldnames = fieldnames
    
    def log(self, metrics: Dict):
        """Log metrics to CSV"""
        if not self.initialized:
            self._initialize(list(metrics.keys()))
        
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            # Convert any non-serializable values
            row = {}
            for k, v in metrics.items():
                if isinstance(v, (dict, list)):
                    row[k] = json.dumps(v)
                elif isinstance(v, float):
                    row[k] = f"{v:.6f}"
                else:
                    row[k] = v
            writer.writerow(row)


# =============== ENHANCED CALLBACKS ===============
class EnhancedCallbacks:
    """Enhanced callbacks with per-class monitoring"""
    
    def __init__(self, save_dir: str, model_name: str, patience: int = 15,
                 min_delta: float = 1e-4, mode: str = 'multiclass', 
                 num_classes: int = 5, class_names: Optional[List[str]] = None,
                 save_csv: bool = True):
        
        self.save_dir = save_dir
        self.model_name = model_name
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.best_loss = float('inf')
        self.best_iou = 0.0
        self.best_f1 = 0.0
        self.best_per_class_iou = None
        self.counter = 0
        self.early_stop = False
        
        # CSV logger
        self.csv_logger = CSVLogger(save_dir, model_name, num_classes, class_names) if save_csv else None
    
    def on_epoch_end(self, epoch: int, model: nn.Module, val_loss: float,
                     val_iou: float, val_f1: float, per_class_metrics: Optional[Dict],
                     optimizer, scheduler) -> bool:
        """Check for improvements and save checkpoints"""
        
        improvement_found = False
        
        # Check loss improvement
        if val_loss < self.best_loss - self.min_delta:
            print(f"  ✓ New best loss: {val_loss:.4f} (prev: {self.best_loss:.4f})")
            self.best_loss = val_loss
            improvement_found = True
            self._save_checkpoint(model, optimizer, scheduler, epoch, 'best_loss', val_loss, val_iou, val_f1)
        
        # Check IoU improvement
        if val_iou > self.best_iou + self.min_delta:
            metric_name = 'IoU' if self.mode == 'binary' else 'mIoU'
            print(f"  ✓ New best {metric_name}: {val_iou:.4f} (prev: {self.best_iou:.4f})")
            self.best_iou = val_iou
            improvement_found = True
            self._save_checkpoint(model, optimizer, scheduler, epoch, 'best_iou', val_loss, val_iou, val_f1)
        
        # Check F1 improvement
        if val_f1 > self.best_f1 + self.min_delta:
            print(f"  ✓ New best F1: {val_f1:.4f} (prev: {self.best_f1:.4f})")
            self.best_f1 = val_f1
            improvement_found = True
        
        # Save combined checkpoint
        if improvement_found:
            self.counter = 0
            self._save_checkpoint(model, optimizer, scheduler, epoch, 'best_combined', val_loss, val_iou, val_f1, per_class_metrics)
        else:
            self.counter += 1
            print(f"  No improvement - Patience: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"  Early stopping triggered after {epoch+1} epochs")
        
        return self.early_stop
    
    def _save_checkpoint(self, model, optimizer, scheduler, epoch, name, 
                         val_loss, val_iou, val_f1, per_class_metrics=None):
        """Save model checkpoint"""
        checkpoint = {
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
            'per_class_metrics': per_class_metrics,
            'mode': self.mode
        }
        
        path = os.path.join(self.save_dir, f"{self.model_name}_{name}.pth")
        torch.save(checkpoint, path)


# =============== DATA AUGMENTATION ===============
class MultiChannelAugmentation:
    """Advanced data augmentation for multi-channel satellite imagery"""
    
    def __init__(self, n_channels: int, augmentation_prob: float = 0.8,
                 geometric_aug: bool = True, noise_aug: bool = True,
                 brightness_aug: bool = True, channel_group_aug: bool = False):
        
        self.n_channels = n_channels
        self.augmentation_prob = augmentation_prob
        self.geometric_aug = geometric_aug
        self.noise_aug = noise_aug
        self.brightness_aug = brightness_aug
        self.channel_group_aug = channel_group_aug
        
        self.channel_groups = {}
        if n_channels >= 10 and channel_group_aug:
            self.channel_groups = {
                'visible': [0, 1, 2, 3],
                'red_edge': [4, 5, 6],
                'nir': [7, 8],
                'swir': [9] if n_channels > 9 else []
            }
    
    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.augmentation_prob:
            return img, mask
        
        try:
            # Geometric augmentations
            if self.geometric_aug:
                if random.random() < 0.5:
                    img = torch.flip(img, dims=[2])
                    mask = torch.flip(mask, dims=[-1])
                if random.random() < 0.5:
                    img = torch.flip(img, dims=[1])
                    mask = torch.flip(mask, dims=[-2] if mask.dim() >= 2 else [-1])
                if random.random() < 0.5:
                    angle = random.choice([90, 180, 270])
                    img = self._rotate_discrete(img, angle)
                    mask = self._rotate_discrete(mask, angle)
            
            # Noise augmentation
            if self.noise_aug and random.random() < 0.3:
                noise = torch.randn_like(img) * 0.03
                img = img + noise
            
            # Brightness augmentation
            if self.brightness_aug and random.random() < 0.4:
                factor = random.uniform(0.85, 1.15)
                img = img * factor
            
            img = torch.clamp(img, 0, 1)
            
        except Exception as e:
            warnings.warn(f"Augmentation error: {e}")
        
        return img, mask
    
    def _rotate_discrete(self, tensor: torch.Tensor, angle: int) -> torch.Tensor:
        if angle == 90:
            return tensor.transpose(-2, -1).flip(-2)
        elif angle == 180:
            return tensor.flip(-2).flip(-1)
        elif angle == 270:
            return tensor.transpose(-2, -1).flip(-1)
        return tensor


# =============== DATASET ===============
class UnifiedSegmentationDataset(Dataset):
    """Dataset for segmentation"""
    
    def __init__(self, folder: str, image_subdir: str = 'images',
                 mask_subdir: str = 'labels', transform: Optional[Callable] = None,
                 mode: str = 'multiclass', num_classes: int = 5):
        
        self.image_dir = Path(folder) / image_subdir
        self.mask_dir = Path(folder) / mask_subdir
        self.transform = transform
        self.mode = mode
        self.num_classes = num_classes
        
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
        
        self.images = sorted([p for p in self.image_dir.iterdir() if p.is_file() and p.suffix in valid_extensions])
        self.masks = sorted([p for p in self.mask_dir.iterdir() if p.is_file() and p.suffix in valid_extensions])
        
        print(f"Found files:")
        print(f"  - Images: {len(self.images)} files")
        print(f"  - Masks: {len(self.masks)} files")
        
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")
        if len(self.masks) == 0:
            raise RuntimeError(f"No masks found in {self.mask_dir}")
    
    def __len__(self) -> int:
        return min(len(self.images), len(self.masks))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        img_stem = img_path.stem
        
        mask_path = None
        for mask in self.masks:
            if mask.stem == img_stem:
                mask_path = mask
                break
        
        if mask_path is None:
            raise RuntimeError(f"Mask not found for {img_path.name}")
        
        img = tiff.imread(img_path).astype(np.float32)
        mask = tiff.imread(mask_path)
        
        # Normalize
        if np.max(img) > 0:
            p99 = np.percentile(img, 99)
            img = np.clip(img / p99, 0, 1)
        
        # Handle mask
        if self.mode == 'binary':
            mask = (mask > 0).astype(np.float32)[np.newaxis, ...]
        else:
            mask = np.clip(mask.astype(np.int64), 0, self.num_classes - 1)
        
        # Handle channels
        if img.ndim == 2:
            img = img[..., np.newaxis]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        if self.mode == 'binary':
            mask = torch.from_numpy(mask).float()
        else:
            mask = torch.from_numpy(mask).long()
        
        if self.transform:
            img, mask = self.transform(img, mask)
        
        return img, mask


# =============== MODEL FACTORY ===============
class ModelFactory:
    """Factory for building segmentation models"""
    
    @staticmethod
    def build_model(name: str, in_channels: int = 3, classes: int = 5,
                    encoder_name: str = 'resnet34', pretrained: bool = True,
                    dropout_rate: float = 0.3, mode: str = 'multiclass') -> nn.Module:
        
        name = name.lower()
        actual_classes = 1 if mode == 'binary' else classes
        
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
                    classes=actual_classes,
                    encoder_weights='imagenet' if pretrained else None,
                    activation=None
                )
        
        raise ValueError(f"Unknown model: {name}")


# =============== MAIN TRAINING FUNCTION ===============
def train_model_v2(config: TrainingConfig, train_loader: DataLoader, 
                   val_loader: DataLoader, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Main training loop with all enhancements"""
    
    print(f"\n{'='*60}")
    print("TRAINING WITH ENHANCED FEATURES V2")
    print(f"{'='*60}")
    
    # Build model
    model = ModelFactory.build_model(
        config.model_name, config.in_channels, config.num_classes,
        config.encoder_name, config.pretrained, config.dropout_rate, config.mode
    )
    model.to(device)
    
    # Calculate class weights
    class_weights = None
    if config.use_class_weights:
        class_weights = calculate_class_weights(train_loader, device, config.mode, config.num_classes)
    
    # Create loss function
    criterion = LossFactory.create_loss(config, class_weights)
    print(f"\nLoss function: {config.loss_type}")
    
    # Encoder freezing
    encoder_freezer = None
    if config.freeze_encoder:
        encoder_freezer = EncoderFreezer(model, config.freeze_epochs)
        encoder_freezer.freeze()
        print(f"Encoder will be frozen for {config.freeze_epochs} epochs")
    
    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Scheduler
    if config.scheduler_type == 'reduce_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    elif config.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate, 
                                                         epochs=config.epochs, steps_per_epoch=len(train_loader))
    
    # Warmup
    warmup_scheduler = None
    if config.warmup_epochs > 0:
        warmup_scheduler = WarmupScheduler(optimizer, config.warmup_epochs, config.warmup_lr,
                                            config.learning_rate, len(train_loader))
        print(f"Learning rate warmup: {config.warmup_epochs} epochs")
    
    # Mixed precision
    scaler = GradScaler() if config.use_amp else None
    if config.use_amp:
        print("Mixed precision training: Enabled")
    
    # Callbacks
    callbacks = EnhancedCallbacks(
        config.save_dir, config.model_name, config.patience, config.min_delta,
        config.mode, config.num_classes, config.class_names, config.save_csv_logs
    )
    
    # History
    history = {
        'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': [],
        'val_accuracy': [], 'learning_rate': [], 'epoch_time': []
    }
    
    # Per-class history
    if config.log_per_class_metrics and config.mode == 'multiclass':
        for i in range(config.num_classes):
            history[f'val_iou_class_{i}'] = []
    
    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"{'='*60}\n")
    
    for epoch in range(config.epochs):
        start_time = time.perf_counter()
        
        # Check if we should unfreeze encoder
        if encoder_freezer:
            if encoder_freezer.check_epoch(epoch):
                # Re-create optimizer with all parameters
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate * 0.1, 
                                               weight_decay=config.weight_decay)
                if config.scheduler_type == 'reduce_plateau':
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        
        # Training phase
        model.train()
        train_loss = 0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            if config.use_amp:
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
            # Warmup step
            if warmup_scheduler and not warmup_scheduler.is_warmup_done():
                warmup_scheduler.step()
        
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        val_metrics = evaluate_model_v2(model, val_loader, device, criterion, config)
        
        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if config.scheduler_type == 'reduce_plateau':
            scheduler.step(val_metrics['loss'])
        elif config.scheduler_type != 'one_cycle':
            scheduler.step()
        
        # Update history
        epoch_time = time.perf_counter() - start_time
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_iou'].append(val_metrics['mean_iou'])
        history['val_f1'].append(val_metrics['mean_f1'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['learning_rate'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        
        # Per-class history
        per_class_metrics = None
        if config.log_per_class_metrics and config.mode == 'multiclass':
            per_class_metrics = val_metrics.get('per_class_iou', {})
            for i in range(config.num_classes):
                iou_val = per_class_metrics.get(f'iou_class_{i}', 0.0)
                history[f'val_iou_class_{i}'].append(iou_val if not np.isnan(iou_val) else 0.0)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config.epochs}:")
        print(f"  Loss: train={train_loss:.4f}, val={val_metrics['loss']:.4f}")
        print(f"  mIoU: {val_metrics['mean_iou']:.4f}, F1: {val_metrics['mean_f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Print per-class IoU
        if config.log_per_class_metrics and config.mode == 'multiclass' and per_class_metrics:
            iou_str = "  Per-class IoU: "
            for i in range(config.num_classes):
                class_name = config.class_names[i] if config.class_names else f"C{i}"
                iou_val = per_class_metrics.get(f'iou_class_{i}', 0.0)
                if np.isnan(iou_val):
                    iou_str += f"{class_name}=N/A "
                else:
                    iou_str += f"{class_name}={iou_val:.3f} "
            print(iou_str)
        
        print(f"  LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")
        
        # Log to CSV
        if callbacks.csv_logger:
            log_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_iou': val_metrics['mean_iou'],
                'val_f1': val_metrics['mean_f1'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            }
            if per_class_metrics:
                for i in range(config.num_classes):
                    log_data[f'iou_class_{i}'] = per_class_metrics.get(f'iou_class_{i}', 0.0)
            callbacks.csv_logger.log(log_data)
        
        # Callbacks
        early_stop = callbacks.on_epoch_end(
            epoch, model, val_metrics['loss'], val_metrics['mean_iou'],
            val_metrics['mean_f1'], per_class_metrics, optimizer, scheduler
        )
        
        print()  # Empty line between epochs
        
        if early_stop:
            print("Early stopping triggered!")
            break
    
    return model, history


def evaluate_model_v2(model: nn.Module, dataloader: DataLoader, device: torch.device,
                      criterion: nn.Module, config: TrainingConfig) -> Dict:
    """Evaluate model with per-class metrics"""
    
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            if config.use_amp:
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            
            total_loss += loss.item() * imgs.size(0)
            all_outputs.append(outputs.cpu())
            all_targets.append(masks.cpu())
    
    mean_loss = total_loss / len(dataloader.dataset)
    outputs = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    metrics = MetricsFactory.compute_metrics(
        config.mode, outputs, targets, config.num_classes,
        compute_per_class=config.log_per_class_metrics
    )
    
    result = {
        'loss': mean_loss,
        'mean_iou': metrics.get('mean_iou', metrics.get('iou_macro', metrics.get('iou', 0))),
        'mean_f1': metrics.get('f1_macro', metrics.get('f1', 0)),
        'accuracy': metrics.get('accuracy_macro', metrics.get('accuracy', 0))
    }
    
    if 'per_class_iou' in metrics:
        result['per_class_iou'] = metrics['per_class_iou']
    if 'per_class_f1' in metrics:
        result['per_class_f1'] = metrics['per_class_f1']
    
    return result


def calculate_class_weights(dataloader: DataLoader, device: torch.device,
                            mode: str, num_classes: int) -> Optional[torch.Tensor]:
    """Calculate class weights for handling imbalance"""
    
    if mode == 'binary':
        total_pixels = 0
        positive_pixels = 0
        
        for _, masks in dataloader:
            positive_pixels += (masks > 0).sum().item()
            total_pixels += masks.numel()
        
        negative_pixels = total_pixels - positive_pixels
        positive_weight = negative_pixels / (positive_pixels + 1e-6)
        
        print(f"\nClass statistics (binary):")
        print(f"  - Positive: {positive_pixels:,} ({100*positive_pixels/total_pixels:.2f}%)")
        print(f"  - Negative: {negative_pixels:,} ({100*negative_pixels/total_pixels:.2f}%)")
        print(f"  - Weight: {positive_weight:.2f}")
        
        return torch.tensor([positive_weight], device=device)
    
    else:
        class_counts = torch.zeros(num_classes, device=device)
        total_pixels = 0
        
        for _, masks in dataloader:
            masks = masks.to(device)
            for cls in range(num_classes):
                class_counts[cls] += (masks == cls).sum().item()
            total_pixels += masks.numel()
        
        # Inverse frequency weighting
        class_weights = total_pixels / (num_classes * class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * num_classes
        
        print(f"\nClass statistics (multi-class):")
        for cls in range(num_classes):
            pct = 100 * class_counts[cls] / total_pixels
            print(f"  - Class {cls}: {class_counts[cls]:,.0f} px ({pct:.2f}%), weight: {class_weights[cls]:.2f}")
        
        return class_weights


# =============== VISUALIZATION ===============
def plot_training_history_v2(history: Dict, save_path: str, config: TrainingConfig):
    """Plot training history with per-class metrics"""
    
    num_classes = config.num_classes
    has_per_class = any(f'val_iou_class_0' in history for _ in [1])
    
    if config.mode == 'multiclass' and has_per_class:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss
        axes[0, 0].plot(history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Global mIoU
        axes[0, 1].plot(history['val_iou'], 'g-', label='mIoU', linewidth=2)
        axes[0, 1].set_title('Mean IoU', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 & Accuracy
        axes[0, 2].plot(history['val_f1'], 'r-', label='F1', linewidth=2)
        axes[0, 2].plot(history['val_accuracy'], 'b-', label='Accuracy', linewidth=2)
        axes[0, 2].set_title('F1 & Accuracy', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Per-class IoU
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        for i in range(num_classes):
            key = f'val_iou_class_{i}'
            if key in history:
                class_name = config.class_names[i] if config.class_names else f'Class {i}'
                axes[1, 0].plot(history[key], color=colors[i], label=class_name, linewidth=2)
        axes[1, 0].set_title('Per-Class IoU', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(history['learning_rate'], 'm-', linewidth=2)
        axes[1, 1].set_title('Learning Rate', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Final per-class IoU bar chart
        final_ious = [history[f'val_iou_class_{i}'][-1] if f'val_iou_class_{i}' in history else 0 
                      for i in range(num_classes)]
        class_labels = config.class_names if config.class_names else [f'C{i}' for i in range(num_classes)]
        bars = axes[1, 2].bar(class_labels, final_ious, color=colors)
        axes[1, 2].set_title('Final Per-Class IoU', fontsize=12, fontweight='bold')
        axes[1, 2].set_ylabel('IoU')
        axes[1, 2].axhline(y=np.mean(final_ious), color='red', linestyle='--', label=f'Mean: {np.mean(final_ious):.3f}')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, val in zip(bars, final_ious):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot(history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(history['val_loss'], 'r-', label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(history['val_iou'], 'g-')
        axes[0, 1].set_title('IoU')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(history['val_f1'], 'r-', label='F1')
        axes[1, 0].plot(history['val_accuracy'], 'b-', label='Accuracy')
        axes[1, 0].set_title('F1 & Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(history['learning_rate'], 'm-')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved: {save_path}")


# =============== MAIN ENTRY POINT ===============
def train_and_save_model_v2(
    model_name: str,
    dataset_root: str,
    mode: str = 'multiclass',
    num_classes: int = 5,
    device: str = 'cuda',
    batch_size: int = 8,
    epochs: int = 100,
    save_dir: str = './trained_models',
    encoder_name: str = 'resnet34',
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    learning_rate: float = 5e-4,
    in_channels: int = 3,
    patch_size: int = 224,
    data_augmentation: bool = True,
    use_class_weights: bool = True,
    patch_subdir: str = 'Patch',
    # New V2 options
    loss_type: str = 'dice_ce',
    freeze_encoder: bool = False,
    freeze_epochs: int = 5,
    warmup_epochs: int = 0,
    use_amp: bool = False,
    log_per_class_metrics: bool = True,
    class_names: Optional[List[str]] = None,
    focal_gamma: float = 2.0,
    tversky_alpha: float = 0.7,
    **kwargs
) -> Dict:
    """Main training function with all V2 features"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    
    # Create config
    config = TrainingConfig(
        mode=mode,
        num_classes=num_classes,
        in_channels=in_channels,
        patch_size=patch_size,
        model_name=model_name,
        encoder_name=encoder_name,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        tversky_alpha=tversky_alpha,
        use_class_weights=use_class_weights,
        freeze_encoder=freeze_encoder,
        freeze_epochs=freeze_epochs,
        warmup_epochs=warmup_epochs,
        use_amp=use_amp,
        data_augmentation=data_augmentation,
        log_per_class_metrics=log_per_class_metrics,
        class_names=class_names,
        save_dir=save_dir
    )
    
    # Print configuration
    print("\n" + "="*70)
    print("UNIFIED SEGMENTATION TRAINING SYSTEM V2")
    print("="*70)
    print(f"Mode: {mode.upper()}, Classes: {num_classes}")
    print(f"Model: {model_name} (encoder: {encoder_name})")
    print(f"Loss: {loss_type}")
    print(f"Freeze encoder: {freeze_encoder} ({freeze_epochs} epochs)" if freeze_encoder else "Freeze encoder: No")
    print(f"Warmup: {warmup_epochs} epochs" if warmup_epochs > 0 else "Warmup: No")
    print(f"AMP: {'Yes' if use_amp else 'No'}")
    print(f"Per-class logging: {'Yes' if log_per_class_metrics else 'No'}")
    print("="*70)
    
    # Create augmentation
    augmentation = None
    if data_augmentation:
        augmentation = MultiChannelAugmentation(in_channels, augmentation_prob=0.8)
        print("Data augmentation: Enabled")
    else:
        print("Data augmentation: Disabled")
    
    # Load datasets
    train_path = os.path.join(dataset_root, patch_subdir, 'train')
    val_path = os.path.join(dataset_root, patch_subdir, 'validation')
    test_path = os.path.join(dataset_root, patch_subdir, 'test')
    
    train_dataset = UnifiedSegmentationDataset(train_path, transform=augmentation, mode=mode, num_classes=num_classes)
    val_dataset = UnifiedSegmentationDataset(val_path, mode=mode, num_classes=num_classes)
    test_dataset = UnifiedSegmentationDataset(test_path, mode=mode, num_classes=num_classes)
    
    effective_batch_size = min(batch_size, len(train_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
    
    print(f"\nDataset: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # Train
    model, history = train_model_v2(config, train_loader, val_loader, device)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model_v2(model, test_loader, device, 
                                      LossFactory.create_loss(config, None), config)
    
    # Save final model
    model_save_path = os.path.join(save_dir, f"{model_name}_{mode}_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': asdict(config),
        'test_metrics': test_metrics,
        'history': history
    }, model_save_path)
    print(f"\nFinal model saved: {model_save_path}")
    
    # Plot history
    plot_save_path = os.path.join(save_dir, f"{model_name}_{mode}_training_plot.png")
    plot_training_history_v2(history, plot_save_path, config)
    
    # Print final results
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Test mIoU: {test_metrics['mean_iou']:.4f}")
    print(f"Test F1: {test_metrics['mean_f1']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    if 'per_class_iou' in test_metrics:
        print("\nPer-class Test IoU:")
        for i in range(num_classes):
            class_name = class_names[i] if class_names else f"Class {i}"
            iou = test_metrics['per_class_iou'].get(f'iou_class_{i}', 0)
            print(f"  {class_name}: {iou:.4f}")
    
    return {
        'config': asdict(config),
        'test_metrics': test_metrics,
        'history': history,
        'model_path': model_save_path
    }


# =============== AVAILABLE MODELS ===============
def get_available_models() -> List[str]:
    """Get list of available models"""
    models = []
    if HAS_SMP:
        models.extend(['unet', 'unet++', 'fpn', 'pspnet', 'linknet', 'manet', 'pan', 'deeplabv3', 'deeplabv3+'])
    return models


def get_available_losses() -> List[str]:
    """Get list of available loss functions"""
    return ['ce', 'dice', 'dice_ce', 'focal', 'focal_dice', 'tversky', 'combo']


# =============== CROSS-VALIDATION ===============
class KFoldCrossValidator:
    """K-Fold Cross Validation for segmentation"""
    
    def __init__(self, dataset_root: str, patch_subdir: str = 'Patch',
                 subdirectories: List[str] = None, n_splits: int = 5,
                 random_state: int = 42, data_augmentation: bool = False,
                 in_channels: int = 3, mode: str = 'multiclass', num_classes: int = 5):
        
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
            
            images = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix in valid_extensions])
            masks = sorted([p for p in mask_dir.iterdir() if p.is_file() and p.suffix in valid_extensions])
            
            for img_path in images:
                img_stem = img_path.stem
                mask_path = None
                for mask in masks:
                    if mask.stem == img_stem:
                        mask_path = mask
                        break
                if mask_path:
                    all_pairs.append((img_path, mask_path))
        
        if len(all_pairs) == 0:
            raise RuntimeError("No valid image-mask pairs found.")
        
        print(f"\nTotal pairs collected: {len(all_pairs)}")
        
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
        train_dataset = _DatasetFromPairs(fold['train_pairs'], transform=transform,
                                          mode=self.mode, num_classes=self.num_classes)
        val_dataset = _DatasetFromPairs(fold['val_pairs'], transform=None,
                                        mode=self.mode, num_classes=self.num_classes)
        return train_dataset, val_dataset


class _DatasetFromPairs(Dataset):
    """Internal dataset from explicit pairs (for cross-validation)"""
    
    def __init__(self, pairs: List[Tuple[Path, Path]], transform: Optional[Callable] = None,
                 mode: str = 'multiclass', num_classes: int = 5):
        self.pairs = pairs
        self.transform = transform
        self.mode = mode
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        img = tiff.imread(img_path).astype(np.float32)
        mask = tiff.imread(mask_path)
        
        if np.max(img) > 0:
            p99 = np.percentile(img, 99)
            img = np.clip(img / p99, 0, 1)
        
        if self.mode == 'binary':
            mask = (mask > 0).astype(np.float32)[np.newaxis, ...]
        else:
            mask = np.clip(mask.astype(np.int64), 0, self.num_classes - 1)
        
        if img.ndim == 2:
            img = img[..., np.newaxis]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        if self.mode == 'binary':
            mask = torch.from_numpy(mask).float()
        else:
            mask = torch.from_numpy(mask).long()
        
        if self.transform:
            img, mask = self.transform(img, mask)
        
        return img, mask


def cross_validate_model_v2(
    model_name: str,
    dataset_root: str,
    device: str = 'cuda',
    n_splits: int = 5,
    batch_size: int = 8,
    epochs: int = 100,
    save_dir: str = './saved_models_cv',
    encoder_name: str = 'resnet34',
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    learning_rate: float = 5e-4,
    in_channels: int = 3,
    patch_size: int = 224,
    data_augmentation: bool = True,
    use_class_weights: bool = True,
    patch_subdir: str = 'Patch',
    subdirectories: List[str] = None,
    mode: str = 'multiclass',
    num_classes: int = 5,
    # V2 options
    loss_type: str = 'dice_ce',
    freeze_encoder: bool = False,
    freeze_epochs: int = 5,
    log_per_class_metrics: bool = True,
    class_names: Optional[List[str]] = None,
    **kwargs
) -> Dict:
    """Perform K-Fold Cross Validation with V2 features"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"K-FOLD CROSS VALIDATION V2 (n_splits={n_splits}, mode={mode})")
    print(f"{'='*60}")
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    cv_save_dir = Path(save_dir) / f"cv_{model_name}_{mode}_{timestamp}"
    cv_save_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    augmentation = None
    if data_augmentation:
        augmentation = MultiChannelAugmentation(n_channels=in_channels, augmentation_prob=0.8)
    
    fold_results = []
    all_ious = []
    all_f1s = []
    all_per_class_ious = {i: [] for i in range(num_classes)}
    
    for fold_idx in range(n_splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'='*60}")
        
        train_dataset, val_dataset = cv.get_fold_datasets(fold_idx, transform=augmentation)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
        
        # Create config for this fold
        config = TrainingConfig(
            mode=mode,
            num_classes=num_classes,
            in_channels=in_channels,
            patch_size=patch_size,
            model_name=f"{model_name}_fold{fold_idx}",
            encoder_name=encoder_name,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_type=loss_type,
            use_class_weights=use_class_weights,
            freeze_encoder=freeze_encoder,
            freeze_epochs=freeze_epochs,
            log_per_class_metrics=log_per_class_metrics,
            class_names=class_names,
            save_dir=str(cv_save_dir)
        )
        
        # Train fold
        model, history = train_model_v2(config, train_loader, val_loader, device)
        
        # Get best metrics
        best_iou = max(history['val_iou'])
        best_f1 = max(history['val_f1'])
        
        all_ious.append(best_iou)
        all_f1s.append(best_f1)
        
        # Per-class IoU
        if log_per_class_metrics:
            for i in range(num_classes):
                key = f'val_iou_class_{i}'
                if key in history:
                    best_class_iou = max(history[key])
                    all_per_class_ious[i].append(best_class_iou)
        
        fold_results.append({
            'fold': fold_idx,
            'best_iou': best_iou,
            'best_f1': best_f1,
            'history': history
        })
        
        print(f"\nFold {fold_idx + 1} completed - Best mIoU: {best_iou:.4f}, Best F1: {best_f1:.4f}")
    
    # Compute statistics
    mean_iou = np.mean(all_ious)
    std_iou = np.std(all_ious)
    mean_f1 = np.mean(all_f1s)
    std_f1 = np.std(all_f1s)
    
    # 95% confidence intervals
    ci_iou = scipy.stats.t.interval(0.95, len(all_ious)-1, loc=mean_iou, scale=scipy.stats.sem(all_ious))
    ci_f1 = scipy.stats.t.interval(0.95, len(all_f1s)-1, loc=mean_f1, scale=scipy.stats.sem(all_f1s))
    
    cv_stats = {
        'mean_iou': mean_iou,
        'std_iou': std_iou,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'ci_iou': {'lower': ci_iou[0], 'upper': ci_iou[1]},
        'ci_f1': {'lower': ci_f1[0], 'upper': ci_f1[1]},
        'all_ious': all_ious,
        'all_f1s': all_f1s
    }
    
    # Per-class statistics
    if log_per_class_metrics:
        cv_stats['per_class'] = {}
        for i in range(num_classes):
            if all_per_class_ious[i]:
                class_name = class_names[i] if class_names else f'class_{i}'
                cv_stats['per_class'][class_name] = {
                    'mean_iou': np.mean(all_per_class_ious[i]),
                    'std_iou': np.std(all_per_class_ious[i]),
                    'all_ious': all_per_class_ious[i]
                }
    
    # Save CV results
    cv_results = {
        'cv_stats': cv_stats,
        'fold_results': fold_results,
        'config': {
            'model_name': model_name,
            'mode': mode,
            'num_classes': num_classes,
            'n_splits': n_splits,
            'loss_type': loss_type
        },
        'cv_save_dir': str(cv_save_dir)
    }
    
    # Save JSON
    json_path = cv_save_dir / f"cv_results_{model_name}.json"
    with open(json_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(cv_results, f, indent=2, default=convert)
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION COMPLETED")
    print(f"{'='*60}")
    print(f"\nResults (n={n_splits} folds):")
    print(f"  mIoU: {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"  F1:   {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  95% CI mIoU: [{ci_iou[0]:.4f}, {ci_iou[1]:.4f}]")
    
    if log_per_class_metrics and 'per_class' in cv_stats:
        print(f"\nPer-class mIoU:")
        for class_name, stats in cv_stats['per_class'].items():
            print(f"  {class_name}: {stats['mean_iou']:.4f} ± {stats['std_iou']:.4f}")
    
    print(f"\nResults saved to: {json_path}")
    
    return cv_results


# =============== COMPATIBILITY WITH predict_large_image.py ===============
def build_model_for_prediction(name: str, in_channels: int = 3, classes: int = 5,
                               encoder_name: str = 'resnet34', pretrained: bool = False,
                               dropout_rate: float = 0.3):
    """
    Build model for compatibility with predict_large_image.py
    Maintains the same signature as the original build_model
    """
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


# =============== SAVE METRICS TO JSON ===============
def save_metrics_json(metrics: Dict, save_path: str):
    """Save metrics to JSON file"""
    
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    with open(save_path, 'w') as f:
        json.dump(convert_numpy(metrics), f, indent=2)
    print(f"Metrics saved to: {save_path}")


# =============== ENHANCED train_and_save_model_v2 with JSON export ===============
def train_and_save_model_v2_full(
    model_name: str,
    dataset_root: str,
    mode: str = 'multiclass',
    num_classes: int = 5,
    device: str = 'cuda',
    batch_size: int = 8,
    epochs: int = 100,
    save_dir: str = './trained_models',
    encoder_name: str = 'resnet34',
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    learning_rate: float = 5e-4,
    in_channels: int = 3,
    patch_size: int = 224,
    data_augmentation: bool = True,
    use_class_weights: bool = True,
    patch_subdir: str = 'Patch',
    loss_type: str = 'dice_ce',
    freeze_encoder: bool = False,
    freeze_epochs: int = 5,
    warmup_epochs: int = 0,
    use_amp: bool = False,
    log_per_class_metrics: bool = True,
    class_names: Optional[List[str]] = None,
    focal_gamma: float = 2.0,
    tversky_alpha: float = 0.7,
    **kwargs
) -> Dict:
    """
    Main training function with all V2 features + JSON export
    This is the complete version with all features from V1 + V2 enhancements
    """
    
    # Call the base function
    result = train_and_save_model_v2(
        model_name=model_name,
        dataset_root=dataset_root,
        mode=mode,
        num_classes=num_classes,
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        save_dir=save_dir,
        encoder_name=encoder_name,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        in_channels=in_channels,
        patch_size=patch_size,
        data_augmentation=data_augmentation,
        use_class_weights=use_class_weights,
        patch_subdir=patch_subdir,
        loss_type=loss_type,
        freeze_encoder=freeze_encoder,
        freeze_epochs=freeze_epochs,
        warmup_epochs=warmup_epochs,
        use_amp=use_amp,
        log_per_class_metrics=log_per_class_metrics,
        class_names=class_names,
        focal_gamma=focal_gamma,
        tversky_alpha=tversky_alpha,
        **kwargs
    )
    
    # Save detailed JSON metrics
    json_metrics = {
        'model_name': model_name,
        'mode': mode,
        'encoder_name': encoder_name,
        'in_channels': in_channels,
        'num_classes': num_classes,
        'patch_size': patch_size,
        'loss_type': loss_type,
        'data_augmentation': data_augmentation,
        'use_class_weights': use_class_weights,
        'freeze_encoder': freeze_encoder,
        'freeze_epochs': freeze_epochs if freeze_encoder else 0,
        'warmup_epochs': warmup_epochs,
        'use_amp': use_amp,
        'test_metrics': result['test_metrics'],
        'training_history': result['history'],
        'config': result['config']
    }
    
    json_path = os.path.join(save_dir, f"{model_name}_{mode}_metrics.json")
    save_metrics_json(json_metrics, json_path)
    
    result['json_path'] = json_path
    
    return result


if __name__ == "__main__":
    print("="*50)
    print("SemanticSeg4EO Training System V2")
    print("="*50)
    print(f"\nAvailable models: {get_available_models()}")
    print(f"Available losses: {get_available_losses()}")
    print("\nNew features in V2:")
    print("  - Per-class IoU logging")
    print("  - Focal Loss, Tversky Loss, Combo Loss")
    print("  - Encoder freezing")
    print("  - Learning rate warmup")
    print("  - Mixed precision (AMP)")
    print("  - CSV export")
    print("  - Cross-validation with per-class stats")
