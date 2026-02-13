#!/usr/bin/env python3
"""
MODERN ARCHITECTURES TRAINING SYSTEM v3 + K-FOLD + CONVNEXT
==============================================================
Script d'entraînement optimisé pour données géospatiales (drone, satellite, aérien)
Compatible avec images multi-canaux (1 à N canaux), patches variables (224, 512, etc.)
Supporte binary et multiclass avec validation robuste

AMÉLIORATIONS v3:
- 4 niveaux d'augmentation: basic, advanced, aggressive, extreme
- Augmentations optimisées pour remote sensing / géospatial
- CLI simplifiée
- Losses strictement validées selon le mode (binary/multiclass)
- Dropout configurable dans tous les modèles
- JSON de sortie complet avec historique

NOUVEAU - K-FOLD CROSS-VALIDATION:
- Validation croisée K-Fold intégrée (--use_kfold)
- Nombre de folds configurable (--n_splits)
- Statistiques robustes avec intervalles de confiance 95%
- Métriques par classe et par fold
- Compatible avec tous les niveaux d'augmentation

NOUVEAU - CONVNEXT FAMILY SUPPORT:
- Support complet de la famille ConvNeXt comme encoders
- Variantes: tiny, small, base, large, xlarge
- Compatible avec tous les modèles SMP (UNet, UNet++, MANet, etc.)
- Encoders: convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge

Author: Enhanced version v3 + K-Fold + ConvNeXt
"""

import os
import sys
import time
import json
import csv
import random
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from sklearn.model_selection import KFold
import scipy.stats

import tifffile as tiff

warnings.filterwarnings('ignore')

# Vérifier tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠ tqdm non installé - pip install tqdm pour barre de progression")


# ============================================================================
# VÉRIFICATION DÉPENDANCES
# ============================================================================

def check_dependencies():
    """Vérifie les dépendances disponibles"""
    deps = {}
    
    try:
        import segmentation_models_pytorch as smp
        deps['smp'] = True
        deps['smp_version'] = smp.__version__
    except ImportError:
        deps['smp'] = False
    
    try:
        from transformers import SegformerForSemanticSegmentation
        deps['transformers'] = True
    except ImportError:
        deps['transformers'] = False
    
    try:
        import timm
        deps['timm'] = True
    except ImportError:
        deps['timm'] = False
    
    return deps

DEPS = check_dependencies()


# ============================================================================
# CONVNEXT ENCODERS - LISTE COMPLÈTE (UNET UNIQUEMENT)
# ============================================================================

CONVNEXT_ENCODERS = {
    'convnext_tiny': 'ConvNeXt-Tiny (28M params) - Rapide et efficace',
    'convnext_small': 'ConvNeXt-Small (50M params) - Bon équilibre',
    'convnext_base': 'ConvNeXt-Base (89M params) - Haute performance',
    'convnext_large': 'ConvNeXt-Large (198M params) - État de l\'art',
    'convnext_xlarge': 'ConvNeXt-XLarge (350M params) - Maximum de capacité',
}

# ⚠️ IMPORTANT: ConvNeXt fonctionne UNIQUEMENT avec --model unet
# SMP ne supporte pas ConvNeXt nativement, donc nous utilisons une implémentation custom
# qui est disponible uniquement pour UNet.

RECOMMENDED_ENCODERS = {
    # Classiques (compatibles avec TOUS les modèles SMP)
    'resnet34': 'ResNet-34 (léger, rapide)',
    'resnet50': 'ResNet-50 (équilibré)',
    'resnet101': 'ResNet-101 (puissant)',
    # EfficientNet (compatibles avec TOUS les modèles SMP)
    'efficientnet-b0': 'EfficientNet-B0 (très léger)',
    'efficientnet-b3': 'EfficientNet-B3 (recommandé)',
    'efficientnet-b4': 'EfficientNet-B4 (haute performance)',
    # ConvNeXt (UNIQUEMENT avec --model unet) ✨
    'convnext_tiny': 'ConvNeXt-Tiny (UNet only, moderne, efficace) ⭐',
    'convnext_small': 'ConvNeXt-Small (UNet only, moderne, performant) ⭐⭐',
    'convnext_base': 'ConvNeXt-Base (UNet only, état de l\'art) ⭐⭐⭐',
}


# ============================================================================
# AUGMENTATION LEVELS - OPTIMISÉ GÉOSPATIAL
# ============================================================================

class AugmentationLevel(Enum):
    """Niveaux de data augmentation pour données géospatiales"""
    NONE = "none"
    BASIC = "basic"           # Flip + Rot90 seulement
    ADVANCED = "advanced"     # + Scale, Brightness, Contrast
    AGGRESSIVE = "aggressive" # + Elastic, Blur, Noise, Crop
    EXTREME = "extreme"       # + MixUp, CutMix, Grid distort, tout


def get_augmentation_config(level: AugmentationLevel, in_channels: int = 4) -> 'AugmentationConfig':
    """
    Retourne la configuration d'augmentation selon le niveau.
    Optimisé pour données géospatiales (drone, satellite, aérien).
    """
    config = AugmentationConfig()
    
    if level == AugmentationLevel.NONE:
        config.enabled = False
        return config
    
    # ====== BASIC ======
    # Transformations géométriques simples (invariances naturelles en remote sensing)
    if level in [AugmentationLevel.BASIC, AugmentationLevel.ADVANCED, 
                 AugmentationLevel.AGGRESSIVE, AugmentationLevel.EXTREME]:
        config.enabled = True
        config.prob = 0.8
        # Flips (vue aérienne = pas d'orientation privilégiée)
        config.flip_horizontal = True
        config.flip_horizontal_prob = 0.5
        config.flip_vertical = True
        config.flip_vertical_prob = 0.5
        # Rotation 90° (images carrées, pas d'orientation)
        config.rotation_90 = True
        config.rotation_90_prob = 0.5
    
    # ====== ADVANCED ======
    # + Variations radiométriques (conditions d'éclairage, atmosphère)
    if level in [AugmentationLevel.ADVANCED, AugmentationLevel.AGGRESSIVE, 
                 AugmentationLevel.EXTREME]:
        # Scale (différentes altitudes de vol/zoom)
        config.scale = True
        config.scale_prob = 0.4
        config.scale_min = 0.85
        config.scale_max = 1.15
        # Brightness (variations solaires)
        config.brightness = True
        config.brightness_prob = 0.4
        config.brightness_limit = 0.15
        # Contrast (conditions atmosphériques)
        config.contrast = True
        config.contrast_prob = 0.4
        config.contrast_limit = 0.15
        # Gamma (capteur/exposition)
        config.gamma = True
        config.gamma_prob = 0.3
        config.gamma_min = 0.85
        config.gamma_max = 1.15
        # Channel noise (bruit capteur par bande spectrale)
        config.channel_noise = True
        config.channel_noise_prob = 0.2
        config.channel_noise_std = 0.02
    
    # ====== AGGRESSIVE ======
    # + Déformations et dégradations réalistes
    if level in [AugmentationLevel.AGGRESSIVE, AugmentationLevel.EXTREME]:
        # Random crop (simule différents cadrages)
        config.random_crop = True
        config.random_crop_prob = 0.4
        config.random_crop_scale_min = 0.75
        config.random_crop_scale_max = 1.0
        # Elastic deformation (distorsions terrain/optique)
        config.elastic = True
        config.elastic_prob = 0.25
        config.elastic_alpha = 100.0
        config.elastic_sigma = 10.0
        # Gaussian blur (focus/atmosphère)
        config.gaussian_blur = True
        config.gaussian_blur_prob = 0.25
        config.gaussian_blur_sigma_min = 0.3
        config.gaussian_blur_sigma_max = 1.5
        # Gaussian noise global (bruit capteur)
        config.gaussian_noise = True
        config.gaussian_noise_prob = 0.25
        config.gaussian_noise_std = 0.025
        # Coarse dropout (occlusions: nuages, ombres)
        config.coarse_dropout = True
        config.coarse_dropout_prob = 0.2
        config.coarse_dropout_max_holes = 5
        config.coarse_dropout_max_height = 40
        config.coarse_dropout_max_width = 40
        # Channel dropout (simule bandes spectrales manquantes)
        if in_channels > 2:
            config.channel_dropout = True
            config.channel_dropout_prob = 0.15
            config.channel_dropout_max_channels = min(2, in_channels - 1)
        # Minority class focus
        config.minority_oversample = True
        config.minority_augment_extra = True
    
    # ====== EXTREME ======
    # Toutes les augmentations possibles
    if level == AugmentationLevel.EXTREME:
        config.prob = 0.9  # Plus agressif
        # Rotation libre
        config.rotation_any = True
        config.rotation_any_prob = 0.3
        config.rotation_any_limit = 45.0
        # Grid distortion
        config.grid_distort = True
        config.grid_distort_prob = 0.2
        config.grid_distort_limit = 0.25
        # Motion blur (mouvement drone/satellite)
        config.motion_blur = True
        config.motion_blur_prob = 0.15
        config.motion_blur_kernel = 5
        # MixUp
        config.mixup = True
        config.mixup_prob = 0.2
        config.mixup_alpha = 0.3
        # CutMix
        config.cutmix = True
        config.cutmix_prob = 0.2
        config.cutmix_alpha = 1.0
        # Channel shuffle (pour multi-spectral)
        if in_channels > 3:
            config.channel_shuffle = True
            config.channel_shuffle_prob = 0.1
    
    return config


# ============================================================================
# CONFIGURATION - AUGMENTATION
# ============================================================================

@dataclass
class AugmentationConfig:
    """Configuration détaillée pour la data augmentation géospatiale"""
    
    # Master switch
    enabled: bool = True
    prob: float = 0.8
    
    # === GEOMETRIC TRANSFORMS ===
    flip_horizontal: bool = False
    flip_horizontal_prob: float = 0.5
    flip_vertical: bool = False
    flip_vertical_prob: float = 0.5
    
    rotation_90: bool = False
    rotation_90_prob: float = 0.5
    rotation_any: bool = False
    rotation_any_prob: float = 0.3
    rotation_any_limit: float = 30.0
    
    scale: bool = False
    scale_prob: float = 0.4
    scale_min: float = 0.8
    scale_max: float = 1.2
    
    random_crop: bool = False
    random_crop_prob: float = 0.5
    random_crop_scale_min: float = 0.7
    random_crop_scale_max: float = 1.0
    
    elastic: bool = False
    elastic_prob: float = 0.3
    elastic_alpha: float = 120.0
    elastic_sigma: float = 12.0
    
    grid_distort: bool = False
    grid_distort_prob: float = 0.2
    grid_distort_limit: float = 0.3
    
    # === PIXEL TRANSFORMS ===
    brightness: bool = False
    brightness_prob: float = 0.4
    brightness_limit: float = 0.15
    
    contrast: bool = False
    contrast_prob: float = 0.4
    contrast_limit: float = 0.15
    
    gamma: bool = False
    gamma_prob: float = 0.3
    gamma_min: float = 0.8
    gamma_max: float = 1.2
    
    gaussian_noise: bool = False
    gaussian_noise_prob: float = 0.3
    gaussian_noise_std: float = 0.03
    
    channel_noise: bool = False
    channel_noise_prob: float = 0.2
    channel_noise_std: float = 0.02
    
    channel_shuffle: bool = False
    channel_shuffle_prob: float = 0.1
    
    channel_dropout: bool = False
    channel_dropout_prob: float = 0.2
    channel_dropout_max_channels: int = 2
    
    # === BLUR ===
    gaussian_blur: bool = False
    gaussian_blur_prob: float = 0.3
    gaussian_blur_sigma_min: float = 0.5
    gaussian_blur_sigma_max: float = 2.0
    
    motion_blur: bool = False
    motion_blur_prob: float = 0.2
    motion_blur_kernel: int = 7
    
    # === DROPOUT / CUTOUT ===
    coarse_dropout: bool = False
    coarse_dropout_prob: float = 0.3
    coarse_dropout_max_holes: int = 8
    coarse_dropout_max_height: int = 32
    coarse_dropout_max_width: int = 32
    coarse_dropout_fill_value: float = 0.0
    
    # === MIXUP / CUTMIX ===
    mixup: bool = False
    mixup_prob: float = 0.3
    mixup_alpha: float = 0.4
    
    cutmix: bool = False
    cutmix_prob: float = 0.3
    cutmix_alpha: float = 1.0
    
    # === MINORITY CLASS FOCUS ===
    minority_oversample: bool = False
    minority_class_idx: int = 1
    minority_augment_extra: bool = False
    minority_extra_prob: float = 0.5


@dataclass 
class TrainingConfig:
    """Configuration complète d'entraînement"""
    
    # Core
    mode: str = 'multiclass'
    num_classes: int = 2
    in_channels: int = 4
    patch_size: int = 224
    
    # Model
    model_name: str = 'segformer-b2'
    encoder_name: str = 'resnet34'
    pretrained: bool = True
    dropout_rate: float = 0.3
    
    # Training
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 20
    min_delta: float = 1e-4
    
    # Loss
    loss_type: str = 'focal_dice'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    dice_weight: float = 0.5
    ce_weight: float = 0.5
    
    # Class weights
    use_class_weights: bool = True
    class_weight_method: str = 'inverse_freq'
    
    # Encoder freezing
    freeze_encoder: bool = False
    freeze_epochs: int = 5
    
    # Learning rate schedule
    scheduler_type: str = 'cosine'
    warmup_epochs: int = 0
    warmup_lr: float = 1e-6
    
    # Mixed precision
    use_amp: bool = True
    
    # Data augmentation
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    augmentation_level: str = 'advanced'  # none, basic, advanced, aggressive, extreme
    
    # Logging
    log_per_class_metrics: bool = True
    save_csv_logs: bool = True
    class_names: Optional[List[str]] = None
    
    # Paths
    save_dir: str = './trained_models_modern'
    
    # Performance
    num_workers: int = 0
    pin_memory: bool = True


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss - MULTICLASS ONLY"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dtype == torch.float32:
            target = target.squeeze(1).long()
        ce_loss = F.cross_entropy(pred, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss - MULTICLASS ONLY"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dtype == torch.float32:
            target = target.squeeze(1).long()
        pred_soft = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DiceCELoss(nn.Module):
    """Dice + CrossEntropy - MULTICLASS ONLY"""
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5, 
                 smooth: float = 1e-6, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss(smooth)
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dtype == torch.float32:
            target = target.squeeze(1).long()
        return self.dice_weight * self.dice(pred, target) + self.ce_weight * self.ce(pred, target)


class TverskyLoss(nn.Module):
    """Tversky Loss - MULTICLASS ONLY"""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dtype == torch.float32:
            target = target.squeeze(1).long()
        pred_soft = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        tp = (pred_soft * target_onehot).sum(dim=(2, 3))
        fp = (pred_soft * (1 - target_onehot)).sum(dim=(2, 3))
        fn = ((1 - pred_soft) * target_onehot).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss - MULTICLASS ONLY"""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, 
                 gamma: float = 0.75, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dtype == torch.float32:
            target = target.squeeze(1).long()
        pred_soft = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        tp = (pred_soft * target_onehot).sum(dim=(2, 3))
        fp = (pred_soft * (1 - target_onehot)).sum(dim=(2, 3))
        fn = ((1 - pred_soft) * target_onehot).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1 - tversky, self.gamma).mean()


# === BINARY LOSSES ===

class BinaryDiceLoss(nn.Module):
    """Dice Loss - BINARY ONLY"""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BinaryFocalLoss(nn.Module):
    """Focal BCE - BINARY ONLY"""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class BinaryDiceBCELoss(nn.Module):
    """Dice + BCE - BINARY ONLY (très efficace)"""
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = BinaryDiceLoss(smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        return self.dice_weight * self.dice(logits, targets) + self.bce_weight * self.bce(logits, targets.float())


class BinaryTverskyLoss(nn.Module):
    """Tversky Loss - BINARY ONLY"""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1 - targets)).sum(dim=(2, 3))
        fn = ((1 - probs) * targets).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()


class BinaryFocalTverskyLoss(nn.Module):
    """Focal Tversky Loss - BINARY ONLY"""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, 
                 gamma: float = 0.75, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1 - targets)).sum(dim=(2, 3))
        fn = ((1 - probs) * targets).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1 - tversky, self.gamma).mean()


class BinaryFocalDiceLoss(nn.Module):
    """Focal + Dice - BINARY ONLY"""
    def __init__(self, focal_alpha: float = 0.75, focal_gamma: float = 2.0,
                 dice_weight: float = 0.5, focal_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = BinaryDiceLoss()
        self.focal = BinaryFocalLoss(focal_alpha, focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.focal_weight * self.focal(logits, targets)


class ComboLoss(nn.Module):
    """Combination of multiple losses"""
    
    def __init__(self, losses: List[nn.Module], weights: List[float]):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = 0
        for loss, w in zip(self.losses, self.weights):
            total = total + w * loss(pred, target)
        return total


# ============================================================================
# LOSS FACTORY - STRICT MODE VALIDATION
# ============================================================================

class LossFactory:
    """Factory pour créer les fonctions de perte avec validation stricte du mode"""
    
    # Losses disponibles par mode
    BINARY_LOSSES = {
        'bce', 'binary_dice', 'binary_focal', 'binary_focal_dice',
        'binary_tversky', 'binary_focal_tversky', 'binary_dice_bce'
    }
    
    MULTICLASS_LOSSES = {
        'ce', 'dice', 'focal', 'tversky', 'focal_tversky',
        'dice_ce', 'focal_dice', 'combo'
    }
    
    # Mapping de conversion automatique
    BINARY_TO_MULTICLASS = {
        'bce': 'ce',
        'binary_dice': 'dice',
        'binary_focal': 'focal',
        'binary_focal_dice': 'focal_dice',
        'binary_tversky': 'tversky',
        'binary_focal_tversky': 'focal_tversky',
        'binary_dice_bce': 'dice_ce'
    }
    
    MULTICLASS_TO_BINARY = {v: k for k, v in BINARY_TO_MULTICLASS.items()}
    
    @staticmethod
    def create(config: TrainingConfig, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        loss_type = config.loss_type.lower()
        
        # === VALIDATION ET CONVERSION AUTOMATIQUE ===
        if config.mode == 'binary':
            if loss_type not in LossFactory.BINARY_LOSSES:
                if loss_type in LossFactory.MULTICLASS_LOSSES:
                    new_loss = LossFactory.MULTICLASS_TO_BINARY.get(loss_type, 'binary_focal_dice')
                    print(f"⚠ Auto-conversion: '{loss_type}' → '{new_loss}' (mode binary)")
                    loss_type = new_loss
                else:
                    print(f"⚠ Loss '{loss_type}' inconnue, utilisation de 'binary_focal_dice'")
                    loss_type = 'binary_focal_dice'
        else:  # multiclass
            if loss_type not in LossFactory.MULTICLASS_LOSSES:
                if loss_type in LossFactory.BINARY_LOSSES:
                    new_loss = LossFactory.BINARY_TO_MULTICLASS.get(loss_type, 'focal_dice')
                    print(f"⚠ Auto-conversion: '{loss_type}' → '{new_loss}' (mode multiclass)")
                    loss_type = new_loss
                else:
                    print(f"⚠ Loss '{loss_type}' inconnue, utilisation de 'focal_dice'")
                    loss_type = 'focal_dice'
        
        # === CRÉATION DES LOSSES BINAIRES ===
        if loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        
        elif loss_type == 'binary_dice':
            return BinaryDiceLoss()
        
        elif loss_type == 'binary_focal':
            return BinaryFocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
        
        elif loss_type == 'binary_focal_dice':
            return BinaryFocalDiceLoss(
                focal_alpha=config.focal_alpha, 
                focal_gamma=config.focal_gamma,
                dice_weight=config.dice_weight,
                focal_weight=config.ce_weight
            )
        
        elif loss_type == 'binary_tversky':
            return BinaryTverskyLoss(alpha=config.tversky_alpha, beta=config.tversky_beta)
        
        elif loss_type == 'binary_focal_tversky':
            return BinaryFocalTverskyLoss(alpha=config.tversky_alpha, beta=config.tversky_beta)
        
        elif loss_type == 'binary_dice_bce':
            return BinaryDiceBCELoss(dice_weight=config.dice_weight, bce_weight=config.ce_weight)
        
        # === CRÉATION DES LOSSES MULTICLASS ===
        elif loss_type == 'ce':
            return nn.CrossEntropyLoss(weight=class_weights)
        
        elif loss_type == 'dice':
            return DiceLoss()
        
        elif loss_type == 'focal':
            return FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma, weight=class_weights)
        
        elif loss_type == 'tversky':
            return TverskyLoss(alpha=config.tversky_alpha, beta=config.tversky_beta)
        
        elif loss_type == 'focal_tversky':
            return FocalTverskyLoss(alpha=config.tversky_alpha, beta=config.tversky_beta)
        
        elif loss_type == 'dice_ce':
            return DiceCELoss(dice_weight=config.dice_weight, ce_weight=config.ce_weight, weight=class_weights)
        
        elif loss_type == 'focal_dice':
            return ComboLoss(
                [FocalLoss(gamma=config.focal_gamma, weight=class_weights), DiceLoss()],
                [config.ce_weight, config.dice_weight]
            )
        
        elif loss_type == 'combo':
            return ComboLoss(
                [nn.CrossEntropyLoss(weight=class_weights), DiceLoss(), 
                 FocalLoss(gamma=config.focal_gamma, weight=class_weights)],
                [0.4, 0.3, 0.3]
            )
        
        else:
            raise ValueError(f"Loss inconnue: {loss_type}")
    
    @staticmethod
    def get_losses_for_mode(mode: str) -> set:
        """Retourne les losses disponibles pour un mode donné"""
        if mode == 'binary':
            return LossFactory.BINARY_LOSSES
        return LossFactory.MULTICLASS_LOSSES


# ============================================================================
# ADVANCED DATA AUGMENTATION
# ============================================================================

class AdvancedMultiChannelAugmentation:
    """Augmentation avancée pour images géospatiales multi-canaux"""
    
    def __init__(self, config: AugmentationConfig, patch_size: int, mode: str = 'multiclass'):
        self.config = config
        self.patch_size = patch_size
        self.mode = mode
        
    def __call__(self, img: torch.Tensor, mask: torch.Tensor, 
                 has_minority_class: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.config.enabled:
            return img, mask
            
        if random.random() > self.config.prob:
            return img, mask
        
        img_np = img.numpy()
        if self.mode == 'binary':
            mask_np = mask.squeeze(0).numpy() if mask.dim() == 3 else mask.numpy()
        else:
            mask_np = mask.numpy()
        
        # === GEOMETRIC TRANSFORMS ===
        if self.config.flip_horizontal and random.random() < self.config.flip_horizontal_prob:
            img_np = np.flip(img_np, axis=2).copy()
            mask_np = np.flip(mask_np, axis=1).copy()
        
        if self.config.flip_vertical and random.random() < self.config.flip_vertical_prob:
            img_np = np.flip(img_np, axis=1).copy()
            mask_np = np.flip(mask_np, axis=0).copy()
        
        if self.config.rotation_90 and random.random() < self.config.rotation_90_prob:
            k = random.choice([1, 2, 3])
            img_np = np.rot90(img_np, k, axes=(1, 2)).copy()
            mask_np = np.rot90(mask_np, k, axes=(0, 1)).copy()
        
        if self.config.rotation_any and random.random() < self.config.rotation_any_prob:
            angle = random.uniform(-self.config.rotation_any_limit, self.config.rotation_any_limit)
            img_np, mask_np = self._rotate_any(img_np, mask_np, angle)
        
        if self.config.scale and random.random() < self.config.scale_prob:
            scale = random.uniform(self.config.scale_min, self.config.scale_max)
            img_np, mask_np = self._scale(img_np, mask_np, scale)
        
        if self.config.random_crop and random.random() < self.config.random_crop_prob:
            img_np, mask_np = self._random_crop(img_np, mask_np)
        
        if self.config.elastic and random.random() < self.config.elastic_prob:
            img_np, mask_np = self._elastic_transform(img_np, mask_np)
        
        if self.config.grid_distort and random.random() < self.config.grid_distort_prob:
            img_np, mask_np = self._grid_distort(img_np, mask_np)
        
        # === PIXEL TRANSFORMS ===
        if self.config.brightness and random.random() < self.config.brightness_prob:
            factor = 1 + random.uniform(-self.config.brightness_limit, self.config.brightness_limit)
            img_np = img_np * factor
        
        if self.config.contrast and random.random() < self.config.contrast_prob:
            factor = 1 + random.uniform(-self.config.contrast_limit, self.config.contrast_limit)
            mean = img_np.mean(axis=(1, 2), keepdims=True)
            img_np = (img_np - mean) * factor + mean
        
        if self.config.gamma and random.random() < self.config.gamma_prob:
            gamma = random.uniform(self.config.gamma_min, self.config.gamma_max)
            img_np = np.power(np.clip(img_np, 0, 1), gamma)
        
        if self.config.gaussian_noise and random.random() < self.config.gaussian_noise_prob:
            noise = np.random.normal(0, self.config.gaussian_noise_std, img_np.shape)
            img_np = img_np + noise
        
        if self.config.channel_noise and random.random() < self.config.channel_noise_prob:
            for c in range(img_np.shape[0]):
                if random.random() < 0.5:
                    noise = np.random.normal(0, self.config.channel_noise_std, img_np.shape[1:])
                    img_np[c] = img_np[c] + noise
        
        if self.config.channel_shuffle and img_np.shape[0] > 3:
            if random.random() < self.config.channel_shuffle_prob:
                perm = np.random.permutation(img_np.shape[0])
                img_np = img_np[perm]
        
        if self.config.channel_dropout and img_np.shape[0] > 1:
            if random.random() < self.config.channel_dropout_prob:
                n_drop = random.randint(1, min(self.config.channel_dropout_max_channels, img_np.shape[0] - 1))
                drop_channels = random.sample(range(img_np.shape[0]), n_drop)
                for c in drop_channels:
                    img_np[c] = 0
        
        # === BLUR ===
        if self.config.gaussian_blur and random.random() < self.config.gaussian_blur_prob:
            sigma = random.uniform(self.config.gaussian_blur_sigma_min, self.config.gaussian_blur_sigma_max)
            for c in range(img_np.shape[0]):
                img_np[c] = gaussian_filter(img_np[c], sigma=sigma)
        
        if self.config.motion_blur and random.random() < self.config.motion_blur_prob:
            img_np = self._motion_blur(img_np)
        
        # === DROPOUT ===
        if self.config.coarse_dropout and random.random() < self.config.coarse_dropout_prob:
            img_np = self._coarse_dropout(img_np)
        
        # === MINORITY CLASS EXTRA ===
        if has_minority_class and self.config.minority_augment_extra:
            if random.random() < self.config.minority_extra_prob:
                if random.random() < 0.5:
                    k = random.choice([1, 2, 3])
                    img_np = np.rot90(img_np, k, axes=(1, 2)).copy()
                    mask_np = np.rot90(mask_np, k, axes=(0, 1)).copy()
        
        # Clip and ensure size
        img_np = np.clip(img_np, 0, 1)
        img_np, mask_np = self._ensure_size(img_np, mask_np)
        
        # Convert back
        img = torch.from_numpy(img_np.copy()).float()
        if self.mode == 'binary':
            mask = torch.from_numpy(mask_np.copy()).float().unsqueeze(0)
        else:
            mask = torch.from_numpy(mask_np.copy()).long()
        
        return img, mask
    
    def _rotate_any(self, img: np.ndarray, mask: np.ndarray, angle: float):
        from scipy.ndimage import rotate
        rotated_img = np.zeros_like(img)
        for c in range(img.shape[0]):
            rotated_img[c] = rotate(img[c], angle, reshape=False, order=1, mode='reflect')
        rotated_mask = rotate(mask, angle, reshape=False, order=0, mode='reflect')
        return rotated_img, rotated_mask
    
    def _scale(self, img: np.ndarray, mask: np.ndarray, scale: float):
        from scipy.ndimage import zoom
        h, w = img.shape[1], img.shape[2]
        scaled_img = zoom(img, (1, scale, scale), order=1)
        scaled_mask = zoom(mask.astype(float), scale, order=0)
        if self.mode != 'binary':
            scaled_mask = scaled_mask.astype(np.int64)
        
        if scale > 1:
            start_h = (scaled_img.shape[1] - h) // 2
            start_w = (scaled_img.shape[2] - w) // 2
            scaled_img = scaled_img[:, start_h:start_h+h, start_w:start_w+w]
            scaled_mask = scaled_mask[start_h:start_h+h, start_w:start_w+w]
        else:
            pad_h = (h - scaled_img.shape[1]) // 2
            pad_w = (w - scaled_img.shape[2]) // 2
            pad_h2 = h - scaled_img.shape[1] - pad_h
            pad_w2 = w - scaled_img.shape[2] - pad_w
            scaled_img = np.pad(scaled_img, ((0, 0), (pad_h, pad_h2), (pad_w, pad_w2)), mode='reflect')
            scaled_mask = np.pad(scaled_mask, ((pad_h, pad_h2), (pad_w, pad_w2)), mode='reflect')
        
        return scaled_img, scaled_mask
    
    def _random_crop(self, img: np.ndarray, mask: np.ndarray):
        from scipy.ndimage import zoom
        h, w = img.shape[1], img.shape[2]
        crop_scale = random.uniform(self.config.random_crop_scale_min, self.config.random_crop_scale_max)
        crop_h, crop_w = int(h * crop_scale), int(w * crop_scale)
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        cropped_img = img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        cropped_mask = mask[start_h:start_h+crop_h, start_w:start_w+crop_w]
        zoom_h, zoom_w = h / crop_h, w / crop_w
        resized_img = zoom(cropped_img, (1, zoom_h, zoom_w), order=1)
        resized_mask = zoom(cropped_mask.astype(float), (zoom_h, zoom_w), order=0)
        if self.mode != 'binary':
            resized_mask = resized_mask.astype(np.int64)
        return resized_img, resized_mask
    
    def _elastic_transform(self, img: np.ndarray, mask: np.ndarray):
        alpha, sigma = self.config.elastic_alpha, self.config.elastic_sigma
        h, w = img.shape[1], img.shape[2]
        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        transformed_img = np.zeros_like(img)
        for c in range(img.shape[0]):
            transformed_img[c] = map_coordinates(img[c], indices, order=1, mode='reflect').reshape(h, w)
        transformed_mask = map_coordinates(mask.astype(float), indices, order=0, mode='reflect').reshape(h, w)
        if self.mode != 'binary':
            transformed_mask = transformed_mask.astype(np.int64)
        return transformed_img, transformed_mask
    
    def _grid_distort(self, img: np.ndarray, mask: np.ndarray):
        from scipy.interpolate import interp1d
        h, w = img.shape[1], img.shape[2]
        num_steps = 4
        distort = self.config.grid_distort_limit
        x_steps = np.linspace(0, w, num_steps + 1)
        y_steps = np.linspace(0, h, num_steps + 1)
        x_offset = np.random.uniform(-distort, distort, (num_steps + 1,)) * (w // num_steps)
        y_offset = np.random.uniform(-distort, distort, (num_steps + 1,)) * (h // num_steps)
        x_offset[0] = x_offset[-1] = 0
        y_offset[0] = y_offset[-1] = 0
        x = np.arange(w)
        y = np.arange(h)
        fx = interp1d(x_steps, x_offset, kind='cubic', fill_value='extrapolate')
        fy = interp1d(y_steps, y_offset, kind='cubic', fill_value='extrapolate')
        map_x = np.tile(x + fx(x), (h, 1))
        map_y = np.tile((y + fy(y)).reshape(-1, 1), (1, w))
        indices = map_y.flatten(), map_x.flatten()
        transformed_img = np.zeros_like(img)
        for c in range(img.shape[0]):
            transformed_img[c] = map_coordinates(img[c], indices, order=1, mode='reflect').reshape(h, w)
        transformed_mask = map_coordinates(mask.astype(float), indices, order=0, mode='reflect').reshape(h, w)
        if self.mode != 'binary':
            transformed_mask = transformed_mask.astype(np.int64)
        return transformed_img, transformed_mask
    
    def _motion_blur(self, img: np.ndarray) -> np.ndarray:
        from scipy.ndimage import convolve
        kernel_size = self.config.motion_blur_kernel
        kernel = np.zeros((kernel_size, kernel_size))
        if random.random() < 0.5:
            kernel[kernel_size // 2, :] = 1 / kernel_size
        else:
            kernel[:, kernel_size // 2] = 1 / kernel_size
        blurred = np.zeros_like(img)
        for c in range(img.shape[0]):
            blurred[c] = convolve(img[c], kernel, mode='reflect')
        return blurred
    
    def _coarse_dropout(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[1], img.shape[2]
        n_holes = random.randint(1, self.config.coarse_dropout_max_holes)
        for _ in range(n_holes):
            hole_h = random.randint(8, self.config.coarse_dropout_max_height)
            hole_w = random.randint(8, self.config.coarse_dropout_max_width)
            y = random.randint(0, max(0, h - hole_h))
            x = random.randint(0, max(0, w - hole_w))
            img[:, y:y+hole_h, x:x+hole_w] = self.config.coarse_dropout_fill_value
        return img
    
    def _ensure_size(self, img: np.ndarray, mask: np.ndarray):
        from scipy.ndimage import zoom
        target = self.patch_size
        h, w = img.shape[1], img.shape[2]
        if h != target or w != target:
            zoom_h, zoom_w = target / h, target / w
            img = zoom(img, (1, zoom_h, zoom_w), order=1)
            mask_float = zoom(mask.astype(float), (zoom_h, zoom_w), order=0)
            if self.mode != 'binary':
                mask = mask_float.astype(np.int64)
            else:
                mask = mask_float
        return img, mask


# ============================================================================
# MIXUP / CUTMIX
# ============================================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y.float() + (1 - lam) * y[index].float()
    return mixed_x, mixed_y, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    W, H = x.size(3), x.size(2)
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    mixed_y = y.clone().float()
    mixed_y[:, bby1:bby2, bbx1:bbx2] = y[index, bby1:bby2, bbx1:bbx2].float()
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return mixed_x, mixed_y, lam


# ============================================================================
# DATASET
# ============================================================================

class SegmentationDataset(Dataset):
    """Dataset pour images géospatiales multi-canaux"""
    
    def __init__(self, root_dir: str, split: str = 'train', patch_subdir: str = 'Patch',
                 in_channels: int = 4, num_classes: int = 2, mode: str = 'multiclass',
                 transform: Optional[Callable] = None, minority_class_idx: int = 1):
        self.root_dir = Path(root_dir)
        self.split = split
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mode = mode
        self.transform = transform
        self.minority_class_idx = minority_class_idx
        
        self.image_dir = self.root_dir / patch_subdir / split / 'images'
        self.label_dir = self.root_dir / patch_subdir / split / 'labels'
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
        
        valid_ext = {'.tif', '.tiff', '.TIF', '.TIFF'}
        self.images = sorted([p for p in self.image_dir.iterdir() if p.suffix in valid_ext])
        self.labels = sorted([p for p in self.label_dir.iterdir() if p.suffix in valid_ext])
        
        self.pairs = []
        for img_path in self.images:
            for lbl_path in self.labels:
                if img_path.stem == lbl_path.stem:
                    self.pairs.append((img_path, lbl_path))
                    break
        
        print(f"  {split}: {len(self.pairs)} pairs found")
        
        self.has_minority_class = []
        self._compute_minority_presence()
    
    def _compute_minority_presence(self):
        print(f"  Analyzing minority class presence...")
        for img_path, lbl_path in self.pairs:
            label = tiff.imread(str(lbl_path))
            has_minority = (label == self.minority_class_idx).any()
            self.has_minority_class.append(has_minority)
        n_with = sum(self.has_minority_class)
        print(f"  {n_with}/{len(self.pairs)} patches contain minority class")
    
    def get_sample_weights(self) -> torch.Tensor:
        weights = []
        n_minority = sum(self.has_minority_class)
        n_majority = len(self.pairs) - n_minority
        if n_minority == 0:
            return torch.ones(len(self.pairs))
        w_minority = n_majority / n_minority if n_minority > 0 else 1.0
        for has_m in self.has_minority_class:
            weights.append(w_minority if has_m else 1.0)
        return torch.tensor(weights)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]
        img = tiff.imread(str(img_path)).astype(np.float32)
        label = tiff.imread(str(lbl_path))
        
        if img.max() > 0:
            p99 = np.percentile(img, 99)
            img = np.clip(img / (p99 + 1e-6), 0, 1)
        
        if self.mode == 'binary':
            label = (label > 0).astype(np.float32)
        else:
            label = np.clip(label.astype(np.int64), 0, self.num_classes - 1)
        
        if img.ndim == 2:
            img = img[..., np.newaxis]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        if self.mode == 'binary':
            label = torch.from_numpy(label).unsqueeze(0).float()
        else:
            label = torch.from_numpy(label).long()
        
        if self.transform:
            has_minority = self.has_minority_class[idx]
            img, label = self.transform(img, label, has_minority)
        
        return img, label


# ============================================================================
# K-FOLD CROSS-VALIDATION CLASSES
# ============================================================================

class _DatasetFromPairs(Dataset):
    """Dataset interne à partir de paires (image, mask) explicites pour K-Fold CV"""
    
    def __init__(self, pairs: List[Tuple[Path, Path]], transform: Optional[Callable] = None,
                 mode: str = 'multiclass', num_classes: int = 5):
        self.pairs = pairs
        self.transform = transform
        self.mode = mode
        self.num_classes = num_classes
        
        # Compute minority class presence
        self.has_minority_class = []
        self._compute_minority_presence()
    
    def _compute_minority_presence(self):
        """Analyse la présence de la classe minoritaire (class 1 par défaut)"""
        for img_path, lbl_path in self.pairs:
            label = tiff.imread(str(lbl_path))
            has_minority = (label == 1).any()
            self.has_minority_class.append(has_minority)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        img = tiff.imread(str(img_path)).astype(np.float32)
        mask = tiff.imread(str(mask_path))
        
        # Normalisation
        if np.max(img) > 0:
            p99 = np.percentile(img, 99)
            img = np.clip(img / (p99 + 1e-6), 0, 1)
        
        # Préparation du mask selon le mode
        if self.mode == 'binary':
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.clip(mask.astype(np.int64), 0, self.num_classes - 1)
        
        # Conversion en tensors
        if img.ndim == 2:
            img = img[..., np.newaxis]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        if self.mode == 'binary':
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).long()
        
        # Augmentation
        if self.transform:
            has_minority = self.has_minority_class[idx]
            img, mask = self.transform(img, mask, has_minority)
        
        return img, mask


class KFoldCrossValidator:
    """Validation croisée K-Fold pour segmentation géospatiale"""
    
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
        """Prépare les splits K-Fold à partir du dataset"""
        all_pairs = []
        
        for subdir in self.subdirectories:
            image_dir = self.dataset_root / self.patch_subdir / subdir / 'images'
            mask_dir = self.dataset_root / self.patch_subdir / subdir / 'labels'
            
            if not image_dir.exists() or not mask_dir.exists():
                warnings.warn(f"Répertoire non trouvé: {image_dir} ou {mask_dir}, ignoré...")
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
            raise RuntimeError("Aucune paire image-mask valide trouvée.")
        
        print(f"\nTotal de paires collectées: {len(all_pairs)}")
        
        # K-Fold split
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
        
        print(f"\nValidation croisée {self.n_splits}-fold préparée:")
        for fold in self.folds:
            print(f"  Fold {fold['fold']}: {fold['train_size']} train, {fold['val_size']} validation")
    
    def get_fold_datasets(self, fold_idx: int, transform: Optional[Callable] = None):
        """Retourne les datasets pour un fold spécifique"""
        if fold_idx >= len(self.folds):
            raise ValueError(f"Fold {fold_idx} non trouvé.")
        
        fold = self.folds[fold_idx]
        train_dataset = _DatasetFromPairs(fold['train_pairs'], transform=transform,
                                          mode=self.mode, num_classes=self.num_classes)
        val_dataset = _DatasetFromPairs(fold['val_pairs'], transform=None,
                                        mode=self.mode, num_classes=self.num_classes)
        return train_dataset, val_dataset


# ============================================================================
# CONVNEXT UNET - IMPLÉMENTATION CUSTOM AVEC TIMM
# ============================================================================

class ConvNeXtUNet(nn.Module):
    """UNet avec encoder ConvNeXt via timm (implémentation custom réelle)"""
    
    CONVNEXT_MODELS = {
        'convnext_tiny': 'convnext_tiny.fb_in22k_ft_in1k',
        'convnext_small': 'convnext_small.fb_in22k_ft_in1k', 
        'convnext_base': 'convnext_base.fb_in22k_ft_in1k',
        'convnext_large': 'convnext_large.fb_in22k_ft_in1k',
        'convnext_xlarge': 'convnext_xlarge.fb_in22k_ft_in1k',
    }
    
    def __init__(self, encoder_name: str, num_classes: int, in_channels: int = 3,
                 pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        
        if not DEPS.get('timm'):
            raise ImportError("ConvNeXt requires timm: pip install timm")
        
        import timm
        
        # Obtenir le nom du modèle timm
        if encoder_name not in self.CONVNEXT_MODELS:
            raise ValueError(f"Encoder {encoder_name} not in {list(self.CONVNEXT_MODELS.keys())}")
        
        timm_model_name = self.CONVNEXT_MODELS[encoder_name]
        
        # Créer l'encoder avec features_only=True
        self.encoder = timm.create_model(
            timm_model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            drop_rate=dropout_rate
        )
        
        # Obtenir les canaux de sortie de chaque stage
        encoder_channels = self.encoder.feature_info.channels()
        
        # Decoder UNet classique
        self.decoder_channels = [256, 128, 64, 32]
        
        # Center block
        self.center = self._conv_block(encoder_channels[-1], self.decoder_channels[0], dropout_rate)
        
        # Decoder blocks avec skip connections
        self.decoder_blocks = nn.ModuleList()
        for i, dec_ch in enumerate(self.decoder_channels):
            if i < len(encoder_channels) - 1:
                enc_ch = encoder_channels[-(i+2)]  # Skip connection
                in_ch = dec_ch + enc_ch
            else:
                in_ch = dec_ch
            
            out_ch = self.decoder_channels[i+1] if i+1 < len(self.decoder_channels) else dec_ch
            self.decoder_blocks.append(self._conv_block(in_ch, out_ch, dropout_rate))
        
        # Final conv
        self.final = nn.Conv2d(self.decoder_channels[-1], num_classes, kernel_size=1)
    
    def _conv_block(self, in_ch, out_ch, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        features = self.encoder(x)  # Liste de features maps
        
        # Center
        x = self.center(features[-1])
        
        # Decoder avec skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Upsample
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Skip connection si disponible
            if i < len(features) - 1:
                skip = features[-(i+2)]
                # Adapter la taille si nécessaire
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            
            # Convolution
            x = decoder_block(x)
        
        # Final upsampling vers la taille originale
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.final(x)
        
        return x


# ============================================================================
# MODERN ARCHITECTURES WITH DROPOUT
# ============================================================================

class SegFormerWrapper(nn.Module):
    """SegFormer B0-B5 avec dropout configurable"""
    
    VARIANTS = {
        'segformer-b0': 'nvidia/segformer-b0-finetuned-ade-512-512',
        'segformer-b1': 'nvidia/segformer-b1-finetuned-ade-512-512',
        'segformer-b2': 'nvidia/segformer-b2-finetuned-ade-512-512',
        'segformer-b3': 'nvidia/segformer-b3-finetuned-ade-512-512',
        'segformer-b4': 'nvidia/segformer-b4-finetuned-ade-512-512',
        'segformer-b5': 'nvidia/segformer-b5-finetuned-ade-512-512',
    }
    
    def __init__(self, variant: str, num_classes: int, in_channels: int, 
                 pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        
        if not DEPS.get('transformers'):
            raise ImportError("pip install transformers")
        
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        if pretrained and variant in self.VARIANTS:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.VARIANTS[variant],
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                hidden_dropout_prob=dropout_rate,
                attention_probs_dropout_prob=dropout_rate
            )
        else:
            config = SegformerConfig(
                num_labels=num_classes, 
                num_channels=in_channels,
                hidden_dropout_prob=dropout_rate,
                attention_probs_dropout_prob=dropout_rate
            )
            self.model = SegformerForSemanticSegmentation(config)
        
        if in_channels != 3 and pretrained:
            self._adapt_input_channels(in_channels)
    
    def _adapt_input_channels(self, in_channels: int):
        old_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                            kernel_size=old_conv.kernel_size,
                            stride=old_conv.stride, padding=old_conv.padding)
        with torch.no_grad():
            if in_channels > 3:
                new_conv.weight[:, :3] = old_conv.weight
                for i in range(3, in_channels):
                    new_conv.weight[:, i] = old_conv.weight[:, i % 3]
            else:
                new_conv.weight = nn.Parameter(old_conv.weight[:, :in_channels])
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias
        self.model.segformer.encoder.patch_embeddings[0].proj = new_conv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        return F.interpolate(outputs.logits, size=x.shape[-2:], mode='bilinear', align_corners=False)


class UNetFormer(nn.Module):
    """UNetFormer avec dropout configurable"""
    
    def __init__(self, num_classes: int, in_channels: int, encoder_name: str = 'resnet18',
                 pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        
        if DEPS.get('timm'):
            import timm
            self.encoder = timm.create_model(encoder_name, pretrained=pretrained, 
                                            features_only=True, in_chans=in_channels)
            self.encoder_channels = self.encoder.feature_info.channels()
        elif DEPS.get('smp'):
            import segmentation_models_pytorch as smp
            aux = smp.Unet(encoder_name=encoder_name, in_channels=in_channels, classes=num_classes,
                          encoder_weights='imagenet' if pretrained else None)
            self.encoder = aux.encoder
            self.encoder_channels = list(aux.encoder.out_channels[1:])
        else:
            raise ImportError("pip install timm or segmentation_models_pytorch")
        
        self.decoder = self._build_decoder()
        self.final_conv = nn.Conv2d(64, num_classes, 1)
    
    def _build_decoder(self):
        decoder_channels = [256, 128, 64, 64]
        layers = nn.ModuleList()
        in_ch = self.encoder_channels[-1]
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = self.encoder_channels[-(i+2)] if i < len(self.encoder_channels)-1 else 0
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.Dropout2d(p=self.dropout_rate),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ))
            in_ch = out_ch
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if isinstance(features, dict):
            features = list(features.values())
        out = features[-1]
        for i, layer in enumerate(self.decoder):
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
            if i < len(features) - 1:
                skip = features[-(i+2)]
                if skip.shape[-2:] != out.shape[-2:]:
                    skip = F.interpolate(skip, size=out.shape[-2:], mode='bilinear')
                out = torch.cat([out, skip], dim=1)
            out = layer(out)
        out = self.final_conv(out)
        return F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)


class HRNetSegmentation(nn.Module):
    """HRNet avec dropout configurable"""
    
    def __init__(self, variant: str, num_classes: int, in_channels: int,
                 pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        if not DEPS.get('timm'):
            raise ImportError("pip install timm")
        import timm
        self.backbone = timm.create_model(variant, pretrained=pretrained, 
                                         features_only=True, in_chans=in_channels)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 256, 256)
            features = self.backbone(dummy)
            total_channels = sum(f.shape[1] for f in features)
        self.head = nn.Sequential(
            nn.Conv2d(total_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        target_size = features[0].shape[-2:]
        fused = []
        for f in features:
            if f.shape[-2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            fused.append(f)
        out = torch.cat(fused, dim=1)
        out = self.head(out)
        return F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)


class SwinUNet(nn.Module):
    """Swin-UNet avec dropout configurable"""
    
    def __init__(self, num_classes: int, in_channels: int, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        if not DEPS.get('timm'):
            raise ImportError("pip install timm")
        import timm
        self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained,
                                        features_only=True, in_chans=in_channels)
        encoder_channels = self.encoder.feature_info.channels()
        self.up4 = nn.ConvTranspose2d(encoder_channels[-1], 256, 2, stride=2)
        self.conv4 = self._conv_block(256 + encoder_channels[-2], 256, dropout_rate)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = self._conv_block(128 + encoder_channels[-3], 128, dropout_rate)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = self._conv_block(64 + encoder_channels[-4], 64, dropout_rate)
        self.final = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=4), nn.Conv2d(32, num_classes, 1))
    
    def _conv_block(self, in_ch, out_ch, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        d4 = self.up4(features[-1])
        d4 = torch.cat([d4, F.interpolate(features[-2], size=d4.shape[-2:], mode='bilinear')], dim=1)
        d4 = self.conv4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, F.interpolate(features[-3], size=d3.shape[-2:], mode='bilinear')], dim=1)
        d3 = self.conv3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, F.interpolate(features[-4], size=d2.shape[-2:], mode='bilinear')], dim=1)
        d2 = self.conv2(d2)
        return F.interpolate(self.final(d2), size=x.shape[-2:], mode='bilinear', align_corners=False)


# ============================================================================
# MODEL FACTORY
# ============================================================================

class ModelFactory:
    """Factory pour créer tous les modèles avec dropout"""
    
    MODERN_MODELS = {
        'segformer-b0': ('segformer', 'b0'), 'segformer-b1': ('segformer', 'b1'),
        'segformer-b2': ('segformer', 'b2'), 'segformer-b3': ('segformer', 'b3'),
        'segformer-b4': ('segformer', 'b4'), 'segformer-b5': ('segformer', 'b5'),
        'unetformer': ('unetformer', None),
        'hrnet-w18': ('hrnet', 'hrnet_w18'), 'hrnet-w32': ('hrnet', 'hrnet_w32'), 'hrnet-w48': ('hrnet', 'hrnet_w48'),
        'swin-unet': ('swin-unet', None),
    }
    SMP_MODELS = ['unet', 'unet++', 'deeplabv3+', 'deeplabv3', 'manet', 'fpn', 'pan', 'pspnet', 'linknet']
    
    @classmethod
    def list_models(cls) -> List[str]:
        models = list(cls.MODERN_MODELS.keys())
        if DEPS.get('smp'):
            models.extend(cls.SMP_MODELS)
        return models
    
    @classmethod
    def create(cls, config: TrainingConfig) -> nn.Module:
        name = config.model_name.lower()
        num_classes = 1 if config.mode == 'binary' else config.num_classes
        
        # DÉTECTION CONVNEXT - Si encoder ConvNeXt, utiliser notre implémentation custom
        convnext_encoders = ['convnext_tiny', 'convnext_small', 'convnext_base', 
                            'convnext_large', 'convnext_xlarge']
        
        if config.encoder_name.lower() in convnext_encoders:
            # ConvNeXt n'est supporté qu'avec UNet (implémentation custom)
            if name in ['unet', 'u-net', 'u_net']:
                print(f"✨ Utilisation de ConvNeXt-UNet custom (encoder: {config.encoder_name})")
                return ConvNeXtUNet(
                    encoder_name=config.encoder_name,
                    num_classes=num_classes,
                    in_channels=config.in_channels,
                    pretrained=config.pretrained,
                    dropout_rate=config.dropout_rate
                )
            else:
                raise ValueError(
                    f"ConvNeXt est actuellement supporté uniquement avec UNet.\n"
                    f"Modèle demandé: {name}\n"
                    f"Utilisez --model unet avec --encoder {config.encoder_name}\n"
                    f"Ou choisissez un encoder SMP standard comme efficientnet-b3"
                )
        
        if name in cls.MODERN_MODELS:
            model_type, variant = cls.MODERN_MODELS[name]
            if model_type == 'segformer':
                return SegFormerWrapper(name, num_classes, config.in_channels, config.pretrained, config.dropout_rate)
            elif model_type == 'unetformer':
                return UNetFormer(num_classes, config.in_channels, config.encoder_name, config.pretrained, config.dropout_rate)
            elif model_type == 'hrnet':
                return HRNetSegmentation(variant, num_classes, config.in_channels, config.pretrained, config.dropout_rate)
            elif model_type == 'swin-unet':
                return SwinUNet(num_classes, config.in_channels, config.pretrained, config.dropout_rate)
        
        # SMP models - avec normalisation flexible des noms
        elif DEPS.get('smp'):
            import segmentation_models_pytorch as smp
            smp_map = {
                'unet': smp.Unet, 
                'unet++': smp.UnetPlusPlus, 
                'deeplabv3+': smp.DeepLabV3Plus,
                'deeplabv3': smp.DeepLabV3, 
                'manet': smp.MAnet, 
                'fpn': smp.FPN,
                'pan': smp.PAN, 
                'pspnet': smp.PSPNet, 
                'linknet': smp.Linknet,
            }
            
            # Table de variantes de noms acceptées
            name_variants = {
                'unet': ['unet', 'u-net', 'u_net'],
                'unet++': ['unet++', 'unetplusplus', 'unet-plus-plus', 'unet_plus_plus', 'unetpp'],
                'deeplabv3+': ['deeplabv3+', 'deeplabv3plus', 'deeplab-v3+', 'deeplab_v3_plus', 
                               'deeplabv3-plus', 'deeplab-v3-plus'],
                'deeplabv3': ['deeplabv3', 'deeplab-v3', 'deeplab_v3'],
                'manet': ['manet', 'ma-net', 'ma_net'],
                'fpn': ['fpn'],
                'pan': ['pan'],
                'pspnet': ['pspnet', 'psp-net', 'psp_net'],
                'linknet': ['linknet', 'link-net', 'link_net'],
            }
            
            # Normaliser l'entrée (minuscules)
            name_lower = name.lower()
            
            # Chercher une correspondance dans les variantes
            for canonical_name, variants in name_variants.items():
                if name_lower in variants:
                    return smp_map[canonical_name](
                        encoder_name=config.encoder_name, 
                        in_channels=config.in_channels,
                        classes=num_classes, 
                        encoder_weights='imagenet' if config.pretrained else None,
                        activation=None
                    )
        
        raise ValueError(f"Unknown model: {name}. Available: {cls.list_models()}")


# ============================================================================
# METRICS
# ============================================================================

class MetricsCalculator:
    @staticmethod
    def compute(pred: torch.Tensor, target: torch.Tensor, num_classes: int, mode: str = 'multiclass') -> Dict:
        if mode == 'binary':
            return MetricsCalculator._binary_metrics(pred, target)
        return MetricsCalculator._multiclass_metrics(pred, target, num_classes)
    
    @staticmethod
    def _binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict:
        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        if pred_bin.dim() == 4 and pred_bin.size(1) == 1:
            pred_bin = pred_bin.squeeze(1)
        
        tp = ((pred_bin == 1) & (target == 1)).sum().float()
        tn = ((pred_bin == 0) & (target == 0)).sum().float()
        fp = ((pred_bin == 1) & (target == 0)).sum().float()
        fn = ((pred_bin == 0) & (target == 1)).sum().float()
        
        eps = 1e-6
        iou_pos = (tp / (tp + fp + fn + eps)).item()
        iou_neg = (tn / (tn + fp + fn + eps)).item()
        precision = (tp / (tp + fp + eps)).item()
        recall = (tp / (tp + fn + eps)).item()
        f1_pos = 2 * precision * recall / (precision + recall + eps)
        precision_neg = (tn / (tn + fn + eps)).item()
        recall_neg = (tn / (tn + fp + eps)).item()
        f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + eps)
        accuracy = ((tp + tn) / (tp + tn + fp + fn + eps)).item()
        
        return {
            'mean_iou': iou_pos, 'mean_f1': f1_pos, 'accuracy': accuracy,
            'precision': precision, 'recall': recall,
            'per_class_iou': {'iou_class_0': iou_neg, 'iou_class_1': iou_pos},
            'per_class_f1': {'f1_class_0': f1_neg, 'f1_class_1': f1_pos},
            'precision_class_0': precision_neg, 'precision_class_1': precision,
            'recall_class_0': recall_neg, 'recall_class_1': recall
        }
    
    @staticmethod
    def _multiclass_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict:
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        if target.dtype == torch.float32:
            target = target.long()
        
        metrics = {'per_class_iou': {}, 'per_class_f1': {}}
        iou_list, f1_list = [], []
        total_correct, total_pixels = 0, target.numel()
        
        for c in range(num_classes):
            pred_c, target_c = (pred == c), (target == c)
            tp = (pred_c & target_c).sum().float()
            fp = (pred_c & ~target_c).sum().float()
            fn = (~pred_c & target_c).sum().float()
            total_correct += tp.item()
            eps = 1e-6
            iou = (tp / (tp + fp + fn + eps)).item()
            precision = (tp / (tp + fp + eps)).item()
            recall = (tp / (tp + fn + eps)).item()
            f1 = 2 * precision * recall / (precision + recall + eps)
            metrics['per_class_iou'][f'iou_class_{c}'] = iou
            metrics['per_class_f1'][f'f1_class_{c}'] = f1
            metrics[f'precision_class_{c}'] = precision
            metrics[f'recall_class_{c}'] = recall
            if (tp + fp + fn) > 0:
                iou_list.append(iou)
                f1_list.append(f1)
        
        metrics['mean_iou'] = np.mean(iou_list) if iou_list else 0
        metrics['mean_f1'] = np.mean(f1_list) if f1_list else 0
        metrics['accuracy'] = total_correct / total_pixels if total_pixels > 0 else 0
        return metrics


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def calculate_class_weights(dataloader: DataLoader, device: torch.device, mode: str, num_classes: int):
    print("\nCalculating class weights...")
    if mode == 'binary':
        total, positive = 0, 0
        for _, masks in dataloader:
            positive += (masks > 0).sum().item()
            total += masks.numel()
        negative = total - positive
        weight = negative / (positive + 1e-6)
        print(f"  Positive: {positive:,} ({100*positive/total:.2f}%), Weight: {weight:.2f}")
        return torch.tensor([weight], device=device)
    else:
        class_counts = torch.zeros(num_classes)
        total = 0
        for _, masks in dataloader:
            if masks.dim() == 4:
                masks = masks.squeeze(1)
            masks = masks.long()
            for c in range(num_classes):
                class_counts[c] += (masks == c).sum().item()
            total += masks.numel()
        weights = total / (num_classes * class_counts + 1e-6)
        weights = weights / weights.sum() * num_classes
        for c in range(num_classes):
            print(f"  Class {c}: {class_counts[c]:,.0f} px ({100*class_counts[c]/total:.2f}%), weight: {weights[c]:.2f}")
        return weights.to(device)


class CSVLogger:
    def __init__(self, save_dir: str, model_name: str, num_classes: int, class_names: List[str]):
        self.path = os.path.join(save_dir, f"{model_name}_training_log.csv")
        self.fieldnames = ['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_f1', 'val_accuracy', 'learning_rate', 'epoch_time']
        for i in range(num_classes):
            self.fieldnames.append(f'iou_{class_names[i] if class_names else f"class_{i}"}')
        with open(self.path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
    
    def log(self, metrics: Dict):
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            row = {k: f"{v:.6f}" if isinstance(v, float) else v for k, v in metrics.items() if k in self.fieldnames}
            writer.writerow(row)


class CheckpointManager:
    def __init__(self, save_dir: str, model_name: str, config: TrainingConfig):
        self.save_dir, self.model_name, self.config = save_dir, model_name, config
        self.best_loss, self.best_iou, self.best_f1 = float('inf'), 0.0, 0.0
        self.patience_counter = 0
    
    def save(self, model, optimizer, scheduler, epoch, metrics, checkpoint_type='latest'):
        checkpoint = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': asdict(self.config), 'metrics': metrics,
            'best_loss': self.best_loss, 'best_iou': self.best_iou, 'best_f1': self.best_f1
        }
        path = os.path.join(self.save_dir, f"{self.model_name}_{checkpoint_type}.pth")
        torch.save(checkpoint, path)
        return path
    
    def check_improvement(self, val_loss, val_iou, val_f1, model, optimizer, scheduler, epoch, metrics):
        improved = False
        if val_loss < self.best_loss - self.config.min_delta:
            print(f"  ★ New best loss: {val_loss:.4f}")
            self.best_loss = val_loss
            self.save(model, optimizer, scheduler, epoch, metrics, 'best_loss')
            improved = True
        if val_iou > self.best_iou + self.config.min_delta:
            print(f"  ★ New best mIoU: {val_iou:.4f}")
            self.best_iou = val_iou
            self.save(model, optimizer, scheduler, epoch, metrics, 'best_iou')
            improved = True
        if val_f1 > self.best_f1 + self.config.min_delta:
            print(f"  ★ New best F1: {val_f1:.4f}")
            self.best_f1 = val_f1
            improved = True
        if improved:
            self.patience_counter = 0
            self.save(model, optimizer, scheduler, epoch, metrics, 'best_combined')
        else:
            self.patience_counter += 1
            print(f"  No improvement - Patience: {self.patience_counter}/{self.config.patience}")
        return improved, self.patience_counter >= self.config.patience


def plot_training_history(history: Dict, config: TrainingConfig, save_path: str, class_names: List[str]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['val_iou'], 'g-', label='mIoU', linewidth=2)
    axes[0, 1].plot(epochs, history['val_f1'], 'm-', label='F1', linewidth=2)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Validation Metrics'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, config.num_classes))
    for i in range(config.num_classes):
        key = f'val_iou_class_{i}'
        if key in history:
            axes[1, 0].plot(epochs, history[key], color=colors[i], label=class_names[i], linewidth=2)
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('Per-class Validation IoU'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, history['learning_rate'], 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule'); axes[1, 1].set_yscale('log'); axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f"{config.model_name} - {config.loss_type} - Aug: {config.augmentation_level}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model(config: TrainingConfig, device: torch.device, dataset_root: str) -> Dict:
    print("\n" + "="*70)
    print("MODERN ARCHITECTURES TRAINING SYSTEM v3")
    print("="*70)
    print(f"\n[Configuration]")
    print(f"  Model: {config.model_name} | Encoder: {config.encoder_name}")
    print(f"  Mode: {config.mode} | Classes: {config.num_classes} | Channels: {config.in_channels}")
    print(f"  Patch: {config.patch_size} | Dropout: {config.dropout_rate}")
    print(f"\n[Training]")
    print(f"  Epochs: {config.epochs} | Batch: {config.batch_size} | LR: {config.learning_rate}")
    print(f"  Loss: {config.loss_type} | Class weights: {config.use_class_weights}")
    print(f"  Augmentation: {config.augmentation_level}")
    print("="*70)
    
    os.makedirs(config.save_dir, exist_ok=True)
    class_names = config.class_names or [f"class_{i}" for i in range(config.num_classes)]
    
    # Dataset & Augmentation
    aug_config = get_augmentation_config(
        AugmentationLevel(config.augmentation_level),
        config.in_channels
    )
    config.augmentation = aug_config
    
    transform = None
    if aug_config.enabled:
        transform = AdvancedMultiChannelAugmentation(aug_config, config.patch_size, config.mode)
    
    train_dataset = SegmentationDataset(dataset_root, 'train', 'Patch', config.in_channels, 
                                        config.num_classes, config.mode, transform, aug_config.minority_class_idx)
    val_dataset = SegmentationDataset(dataset_root, 'validation', 'Patch', config.in_channels,
                                     config.num_classes, config.mode, None)
    test_dataset = SegmentationDataset(dataset_root, 'test', 'Patch', config.in_channels,
                                      config.num_classes, config.mode, None)
    
    sampler = None
    shuffle = True
    if aug_config.enabled and aug_config.minority_oversample:
        weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
        print(f"  Using WeightedRandomSampler for minority oversampling")
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle, sampler=sampler,
                             num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                           num_workers=config.num_workers, pin_memory=config.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=config.pin_memory)
    
    # Class weights
    class_weights = calculate_class_weights(train_loader, device, config.mode, config.num_classes) if config.use_class_weights else None
    
    # Model
    print("\n[Creating model]")
    model = ModelFactory.create(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params/1e6:.2f}M")
    
    # Freeze encoder
    if config.freeze_encoder:
        print(f"  Freezing encoder for {config.freeze_epochs} epochs...")
        for name, param in model.named_parameters():
            if any(x in name.lower() for x in ['encoder', 'backbone', 'segformer.encoder']):
                param.requires_grad = False
    
    # Loss
    criterion = LossFactory.create(config, class_weights)
    print(f"[Loss: {config.loss_type}]")
    
    # Optimizer & Scheduler
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    
    if config.scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    elif config.scheduler_type == 'reduce_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    else:
        scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, epochs=config.epochs, steps_per_epoch=len(train_loader))
    
    scaler = GradScaler() if config.use_amp else None
    csv_logger = CSVLogger(config.save_dir, config.model_name.replace('-', '_'), config.num_classes, class_names) if config.save_csv_logs else None
    checkpoint_mgr = CheckpointManager(config.save_dir, config.model_name.replace('-', '_'), config)
    
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': [], 'val_accuracy': [], 'learning_rate': [], 'epoch_time': []}
    for i in range(config.num_classes):
        history[f'val_iou_class_{i}'] = []
    
    print(f"\n{'='*70}\nStarting training for {config.epochs} epochs...\n{'='*70}\n")
    
    for epoch in range(config.epochs):
        start_time = time.perf_counter()
        
        # Unfreeze
        if config.freeze_encoder and epoch == config.freeze_epochs:
            print(f"\n[Epoch {epoch+1}] Unfreezing encoder...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate * 0.1, weight_decay=config.weight_decay)
            if config.scheduler_type == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs - epoch, eta_min=1e-6)
        
        # Warmup
        if epoch < config.warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = config.learning_rate * (epoch + 1) / config.warmup_epochs
        
        # Train
        model.train()
        train_loss = 0
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]', leave=False, ncols=100) if HAS_TQDM else train_loader
        
        for imgs, masks in train_iter:
            imgs, masks = imgs.to(device), masks.to(device)
            
            # MixUp / CutMix
            if aug_config.enabled and aug_config.mixup and random.random() < aug_config.mixup_prob:
                imgs, masks, _ = mixup_data(imgs, masks, aug_config.mixup_alpha)
            elif aug_config.enabled and aug_config.cutmix and random.random() < aug_config.cutmix_prob:
                imgs, masks, _ = cutmix_data(imgs, masks, aug_config.cutmix_alpha)
            
            optimizer.zero_grad()
            if config.use_amp and scaler:
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
            if HAS_TQDM:
                train_iter.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        val_iter = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Val]', leave=False, ncols=100) if HAS_TQDM else val_loader
        
        with torch.no_grad():
            for imgs, masks in val_iter:
                imgs, masks = imgs.to(device), masks.to(device)
                if config.use_amp:
                    with autocast():
                        outputs = model(imgs)
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
                all_preds.append(outputs.cpu())
                all_targets.append(masks.cpu())
        
        val_loss /= len(val_loader.dataset)
        all_preds, all_targets = torch.cat(all_preds), torch.cat(all_targets)
        metrics = MetricsCalculator.compute(all_preds, all_targets, config.num_classes, config.mode)
        
        # Scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if epoch >= config.warmup_epochs:
            if config.scheduler_type == 'reduce_plateau':
                scheduler.step(metrics['mean_iou'])
            elif config.scheduler_type != 'one_cycle':
                scheduler.step()
        
        # Update history
        epoch_time = time.perf_counter() - start_time
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(metrics['mean_iou'])
        history['val_f1'].append(metrics['mean_f1'])
        history['val_accuracy'].append(metrics['accuracy'])
        history['learning_rate'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        for i in range(config.num_classes):
            history[f'val_iou_class_{i}'].append(metrics['per_class_iou'].get(f'iou_class_{i}', 0))
        
        print(f"\nEpoch {epoch+1}/{config.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  mIoU: {metrics['mean_iou']:.4f} | F1: {metrics['mean_f1']:.4f} | Acc: {metrics['accuracy']:.4f}")
        
        if csv_logger:
            log_metrics = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                          'val_iou': metrics['mean_iou'], 'val_f1': metrics['mean_f1'],
                          'val_accuracy': metrics['accuracy'], 'learning_rate': current_lr, 'epoch_time': epoch_time}
            for i in range(config.num_classes):
                log_metrics[f'iou_{class_names[i]}'] = metrics['per_class_iou'].get(f'iou_class_{i}', 0)
            csv_logger.log(log_metrics)
        
        improved, early_stop = checkpoint_mgr.check_improvement(val_loss, metrics['mean_iou'], metrics['mean_f1'],
                                                                model, optimizer, scheduler, epoch, metrics)
        checkpoint_mgr.save(model, optimizer, scheduler, epoch, metrics, 'latest')
        
        if early_stop:
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break
    
    # Final test
    print(f"\n{'='*70}\nFINAL EVALUATION ON TEST SET\n{'='*70}")
    best_path = os.path.join(config.save_dir, f"{config.model_name.replace('-', '_')}_best_iou.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded best IoU model from epoch {checkpoint['epoch']+1}")
    
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            all_preds.append(outputs.cpu())
            all_targets.append(masks)
    
    all_preds, all_targets = torch.cat(all_preds), torch.cat(all_targets)
    test_metrics = MetricsCalculator.compute(all_preds, all_targets, config.num_classes, config.mode)
    
    print(f"\nTest Results: mIoU={test_metrics['mean_iou']:.4f} | F1={test_metrics['mean_f1']:.4f}")
    for i in range(config.num_classes):
        print(f"  {class_names[i]}: IoU={test_metrics['per_class_iou'].get(f'iou_class_{i}', 0):.4f}")
    
    # Plot & Save
    plot_path = os.path.join(config.save_dir, f"{config.model_name.replace('-', '_')}_training.png")
    plot_training_history(history, config, plot_path, class_names)
    
    # Save comprehensive JSON
    metrics_json = {
        "model_name": config.model_name, "mode": config.mode, "encoder_name": config.encoder_name,
        "in_channels": config.in_channels, "num_classes": config.num_classes, "patch_size": config.patch_size,
        "loss_type": config.loss_type, "dropout_rate": config.dropout_rate,
        "augmentation_level": config.augmentation_level,
        "data_augmentation": config.augmentation.enabled, "use_class_weights": config.use_class_weights,
        "freeze_encoder": config.freeze_encoder, "warmup_epochs": config.warmup_epochs, "use_amp": config.use_amp,
        "test_metrics": {
            "loss": val_loss, "mean_iou": test_metrics['mean_iou'], "mean_f1": test_metrics['mean_f1'],
            "accuracy": test_metrics['accuracy'], "precision": test_metrics.get('precision', 0),
            "recall": test_metrics.get('recall', 0), "per_class_iou": test_metrics.get('per_class_iou', {}),
            "per_class_f1": test_metrics.get('per_class_f1', {})
        },
        "best_val_iou": checkpoint_mgr.best_iou, "best_val_loss": checkpoint_mgr.best_loss,
        "best_val_f1": checkpoint_mgr.best_f1, "training_epochs": len(history['train_loss']),
        "training_history": {
            "train_loss": history['train_loss'], "val_loss": history['val_loss'],
            "val_iou": history['val_iou'], "val_f1": history['val_f1'],
            "val_accuracy": history['val_accuracy'], "learning_rate": history['learning_rate'],
            "epoch_time": history['epoch_time']
        },
        "per_class_iou_history": {f"val_iou_{class_names[i]}": history.get(f'val_iou_class_{i}', []) for i in range(config.num_classes)},
        "config": asdict(config)
    }
    metrics_path = os.path.join(config.save_dir, f"{config.model_name.replace('-', '_')}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2, default=str)
    print(f"\nMetrics saved to: {metrics_path}")
    
    return {'model_path': best_path, 'metrics_path': metrics_path, 'history': history, 'test_metrics': test_metrics, 'config': config}


# ============================================================================
# K-FOLD CROSS-VALIDATION
# ============================================================================

def cross_validate_model_v3(
    config: TrainingConfig,
    device: torch.device,
    dataset_root: str,
    n_splits: int = 5,
    subdirectories: List[str] = None
) -> Dict:
    """
    Validation croisée K-Fold avec TOUTES les fonctionnalités du mode standard.
    
    Inclut:
    - CheckpointManager complet (best_loss, best_iou, best_f1) par fold
    - CSVLogger par fold
    - MetricsCalculator pour métriques complètes (accuracy, precision, recall, per-class)
    - Freeze/Unfreeze encoder
    - Warmup LR
    - Plot training history par fold
    - JSON complet par fold + JSON global CV
    - Calcul des class_weights via calculate_class_weights()
    
    Args:
        config: Configuration d'entraînement
        device: Device (cuda/cpu)
        dataset_root: Racine du dataset
        n_splits: Nombre de folds (défaut: 5)
        subdirectories: Sous-répertoires à utiliser
    
    Returns:
        Dict contenant les résultats de CV avec statistiques complètes
    """
    
    print(f"\n{'='*70}")
    print(f"K-FOLD CROSS VALIDATION V3 (n_splits={n_splits}, mode={config.mode})")
    print(f"{'='*70}")
    print(f"\n[Configuration]")
    print(f"  Model: {config.model_name} | Encoder: {config.encoder_name}")
    print(f"  Mode: {config.mode} | Classes: {config.num_classes} | Channels: {config.in_channels}")
    print(f"  Patch: {config.patch_size} | Dropout: {config.dropout_rate}")
    print(f"\n[Training]")
    print(f"  Epochs: {config.epochs} | Batch: {config.batch_size} | LR: {config.learning_rate}")
    print(f"  Loss: {config.loss_type} | Class weights: {config.use_class_weights}")
    print(f"  Augmentation: {config.augmentation_level}")
    print(f"  Freeze encoder: {config.freeze_encoder} ({config.freeze_epochs} epochs)")
    print(f"  Warmup: {config.warmup_epochs} epochs | AMP: {config.use_amp}")
    print("="*70)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    cv_save_dir = Path(config.save_dir) / f"cv_{config.model_name.replace('-', '_')}_{config.mode}_{timestamp}"
    cv_save_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = config.class_names or [f"class_{i}" for i in range(config.num_classes)]
    
    # Préparer l'augmentation (identique au standard)
    aug_config = get_augmentation_config(
        AugmentationLevel(config.augmentation_level),
        config.in_channels
    )
    config.augmentation = aug_config
    
    augmentation = None
    if aug_config.enabled:
        augmentation = AdvancedMultiChannelAugmentation(aug_config, config.patch_size, config.mode)
    
    # Préparer le cross-validator
    cv = KFoldCrossValidator(
        dataset_root=dataset_root,
        patch_subdir='Patch',
        subdirectories=subdirectories or ['train', 'validation', 'test'],
        n_splits=n_splits,
        random_state=42,
        data_augmentation=True,
        in_channels=config.in_channels,
        mode=config.mode,
        num_classes=config.num_classes
    )
    cv.prepare_folds()
    
    # Stocker les résultats globaux
    fold_results = []
    all_ious = []
    all_f1s = []
    all_losses = []
    all_accuracies = []
    all_per_class_ious = {i: [] for i in range(config.num_classes)}
    all_per_class_f1s = {i: [] for i in range(config.num_classes)}
    
    # ========================================================================
    # BOUCLE SUR CHAQUE FOLD
    # ========================================================================
    for fold_idx in range(n_splits):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'='*70}")
        
        # --- Répertoire du fold ---
        fold_save_dir = cv_save_dir / f"fold_{fold_idx}"
        fold_save_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Datasets pour ce fold ---
        train_dataset, val_dataset = cv.get_fold_datasets(fold_idx, transform=augmentation)
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=config.pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=config.pin_memory
        )
        
        # --- Config spécifique fold (pour sauvegarde dans checkpoint) ---
        fold_config = TrainingConfig(
            model_name=config.model_name, encoder_name=config.encoder_name,
            mode=config.mode, num_classes=config.num_classes,
            in_channels=config.in_channels, patch_size=config.patch_size,
            pretrained=config.pretrained, dropout_rate=config.dropout_rate,
            epochs=config.epochs, batch_size=config.batch_size,
            learning_rate=config.learning_rate, weight_decay=config.weight_decay,
            patience=config.patience, min_delta=config.min_delta,
            loss_type=config.loss_type,
            focal_gamma=config.focal_gamma, focal_alpha=config.focal_alpha,
            tversky_alpha=config.tversky_alpha, tversky_beta=config.tversky_beta,
            dice_weight=config.dice_weight, ce_weight=config.ce_weight,
            use_class_weights=config.use_class_weights,
            freeze_encoder=config.freeze_encoder, freeze_epochs=config.freeze_epochs,
            warmup_epochs=config.warmup_epochs, warmup_lr=config.warmup_lr,
            scheduler_type=config.scheduler_type, use_amp=config.use_amp,
            augmentation=aug_config, augmentation_level=config.augmentation_level,
            num_workers=config.num_workers, class_names=config.class_names,
            save_dir=str(fold_save_dir)
        )
        
        # --- Class weights (via calculate_class_weights, identique au standard) ---
        class_weights = None
        if config.use_class_weights:
            class_weights = calculate_class_weights(train_loader, device, config.mode, config.num_classes)
        
        # --- Modèle ---
        print(f"\n[Construction du modèle {config.model_name}]")
        model = ModelFactory.create(fold_config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {total_params/1e6:.2f}M")
        
        # --- Freeze encoder (identique au standard) ---
        if config.freeze_encoder:
            print(f"  Freezing encoder for {config.freeze_epochs} epochs...")
            for name, param in model.named_parameters():
                if any(x in name.lower() for x in ['encoder', 'backbone', 'segformer.encoder']):
                    param.requires_grad = False
        
        # --- Loss ---
        criterion = LossFactory.create(fold_config, class_weights)
        print(f"[Loss: {config.loss_type}]")
        
        # --- Optimizer & Scheduler ---
        trainable = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
        
        if config.scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
        elif config.scheduler_type == 'reduce_plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
        else:
            scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate,
                                   epochs=config.epochs, steps_per_epoch=len(train_loader))
        
        # --- Utilitaires ---
        scaler = GradScaler() if config.use_amp else None
        csv_logger = CSVLogger(str(fold_save_dir), f"fold_{fold_idx}", config.num_classes, class_names)
        checkpoint_mgr = CheckpointManager(str(fold_save_dir), f"fold_{fold_idx}", fold_config)
        
        # --- Historique ---
        history = {
            'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': [],
            'val_accuracy': [], 'learning_rate': [], 'epoch_time': []
        }
        for i in range(config.num_classes):
            history[f'val_iou_class_{i}'] = []
        
        print(f"\nStarting training for fold {fold_idx + 1} ({config.epochs} epochs max)...\n")
        
        # ==================================================================
        # BOUCLE D'ENTRAÎNEMENT PAR EPOCH (parité standard)
        # ==================================================================
        for epoch in range(config.epochs):
            start_time = time.perf_counter()
            
            # --- Unfreeze encoder après freeze_epochs ---
            if config.freeze_encoder and epoch == config.freeze_epochs:
                print(f"\n[Epoch {epoch+1}] Unfreezing encoder...")
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate * 0.1,
                                              weight_decay=config.weight_decay)
                if config.scheduler_type == 'cosine':
                    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs - epoch, eta_min=1e-6)
            
            # --- Warmup LR ---
            if epoch < config.warmup_epochs:
                for pg in optimizer.param_groups:
                    pg['lr'] = config.learning_rate * (epoch + 1) / config.warmup_epochs
            
            # --- TRAIN ---
            model.train()
            train_loss = 0
            n_samples = 0
            
            iterator = tqdm(train_loader, desc=f"F{fold_idx+1} Ep{epoch+1}/{config.epochs} [Train]", leave=False, ncols=100) if HAS_TQDM else train_loader
            for images, masks in iterator:
                images, masks = images.to(device), masks.to(device)
                
                # MixUp / CutMix (identique au standard)
                if aug_config.enabled and aug_config.mixup and random.random() < aug_config.mixup_prob:
                    images, masks, _ = mixup_data(images, masks, aug_config.mixup_alpha)
                elif aug_config.enabled and aug_config.cutmix and random.random() < aug_config.cutmix_prob:
                    images, masks, _ = cutmix_data(images, masks, aug_config.cutmix_alpha)
                
                optimizer.zero_grad()
                
                if config.use_amp and scaler is not None:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                n_samples += images.size(0)
                
                if HAS_TQDM:
                    iterator.set_postfix({'loss': f'{loss.item():.4f}'})
                
                if config.scheduler_type == 'one_cycle' and epoch >= config.warmup_epochs:
                    scheduler.step()
            
            train_loss /= max(n_samples, 1)
            
            # --- VALIDATION (avec MetricsCalculator, identique au standard) ---
            model.eval()
            val_loss = 0
            n_val_samples = 0
            all_preds_list = []
            all_targets_list = []
            
            val_iter = tqdm(val_loader, desc=f"F{fold_idx+1} Ep{epoch+1}/{config.epochs} [Val]", leave=False, ncols=100) if HAS_TQDM else val_loader
            with torch.no_grad():
                for images, masks in val_iter:
                    images, masks = images.to(device), masks.to(device)
                    if config.use_amp:
                        with autocast():
                            outputs = model(images)
                            loss = criterion(outputs, masks)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)
                    n_val_samples += images.size(0)
                    all_preds_list.append(outputs.cpu())
                    all_targets_list.append(masks.cpu())
            
            val_loss /= max(n_val_samples, 1)
            
            # Métriques complètes via MetricsCalculator (parité standard)
            all_preds_t = torch.cat(all_preds_list)
            all_targets_t = torch.cat(all_targets_list)
            metrics = MetricsCalculator.compute(all_preds_t, all_targets_t, config.num_classes, config.mode)
            
            # --- Scheduler step ---
            current_lr = optimizer.param_groups[0]['lr']
            if epoch >= config.warmup_epochs:
                if config.scheduler_type == 'reduce_plateau':
                    scheduler.step(metrics['mean_iou'])
                elif config.scheduler_type != 'one_cycle':
                    scheduler.step()
            
            # --- Historique ---
            epoch_time = time.perf_counter() - start_time
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_iou'].append(metrics['mean_iou'])
            history['val_f1'].append(metrics['mean_f1'])
            history['val_accuracy'].append(metrics['accuracy'])
            history['learning_rate'].append(current_lr)
            history['epoch_time'].append(epoch_time)
            for i in range(config.num_classes):
                history[f'val_iou_class_{i}'].append(metrics['per_class_iou'].get(f'iou_class_{i}', 0))
            
            # --- Affichage ---
            print(f"\nF{fold_idx+1} Epoch {epoch+1}/{config.epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  mIoU: {metrics['mean_iou']:.4f} | F1: {metrics['mean_f1']:.4f} | Acc: {metrics['accuracy']:.4f}")
            for i in range(config.num_classes):
                iou_i = metrics['per_class_iou'].get(f'iou_class_{i}', 0)
                print(f"    {class_names[i]}: IoU={iou_i:.4f}")
            
            # --- CSV log ---
            if csv_logger:
                log_metrics = {
                    'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                    'val_iou': metrics['mean_iou'], 'val_f1': metrics['mean_f1'],
                    'val_accuracy': metrics['accuracy'], 'learning_rate': current_lr,
                    'epoch_time': epoch_time
                }
                for i in range(config.num_classes):
                    log_metrics[f'iou_{class_names[i]}'] = metrics['per_class_iou'].get(f'iou_class_{i}', 0)
                csv_logger.log(log_metrics)
            
            # --- Checkpoint (best_loss, best_iou, best_f1 + early stopping) ---
            improved, early_stop = checkpoint_mgr.check_improvement(
                val_loss, metrics['mean_iou'], metrics['mean_f1'],
                model, optimizer, scheduler, epoch, metrics
            )
            checkpoint_mgr.save(model, optimizer, scheduler, epoch, metrics, 'latest')
            
            if early_stop:
                print(f"\n⚠ Early stopping at epoch {epoch+1}")
                break
        
        # ==================================================================
        # FIN DU FOLD - Sauvegarde & Bilan
        # ==================================================================
        
        # Charger le meilleur modèle (best_iou) pour les métriques finales
        best_iou_path = os.path.join(str(fold_save_dir), f"fold_{fold_idx}_best_iou.pth")
        best_metrics = metrics  # fallback si fichier absent
        
        if os.path.exists(best_iou_path):
            best_ckpt = torch.load(best_iou_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt['model_state_dict'])
            print(f"\n  Loaded best IoU model from epoch {best_ckpt['epoch']+1}")
            
            # Recalculer les métriques de validation avec le best model
            model.eval()
            all_preds_list = []
            all_targets_list = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    all_preds_list.append(outputs.cpu())
                    all_targets_list.append(masks.cpu())
            all_preds_t = torch.cat(all_preds_list)
            all_targets_t = torch.cat(all_targets_list)
            best_metrics = MetricsCalculator.compute(all_preds_t, all_targets_t, config.num_classes, config.mode)
        
        # Plot training history
        fold_plot_path = os.path.join(str(fold_save_dir), f"fold_{fold_idx}_training.png")
        plot_training_history(history, fold_config, fold_plot_path, class_names)
        print(f"  Plot saved: {fold_plot_path}")
        
        # JSON complet par fold (identique au standard)
        fold_json = {
            "fold": fold_idx,
            "model_name": config.model_name, "mode": config.mode,
            "encoder_name": config.encoder_name,
            "in_channels": config.in_channels, "num_classes": config.num_classes,
            "patch_size": config.patch_size, "loss_type": config.loss_type,
            "dropout_rate": config.dropout_rate,
            "augmentation_level": config.augmentation_level,
            "use_class_weights": config.use_class_weights,
            "freeze_encoder": config.freeze_encoder,
            "warmup_epochs": config.warmup_epochs,
            "use_amp": config.use_amp,
            "val_metrics": {
                "mean_iou": best_metrics['mean_iou'],
                "mean_f1": best_metrics['mean_f1'],
                "accuracy": best_metrics['accuracy'],
                "precision": best_metrics.get('precision', 0),
                "recall": best_metrics.get('recall', 0),
                "per_class_iou": best_metrics.get('per_class_iou', {}),
                "per_class_f1": best_metrics.get('per_class_f1', {})
            },
            "best_val_iou": checkpoint_mgr.best_iou,
            "best_val_loss": checkpoint_mgr.best_loss,
            "best_val_f1": checkpoint_mgr.best_f1,
            "training_epochs": len(history['train_loss']),
            "training_history": {
                "train_loss": history['train_loss'],
                "val_loss": history['val_loss'],
                "val_iou": history['val_iou'],
                "val_f1": history['val_f1'],
                "val_accuracy": history['val_accuracy'],
                "learning_rate": history['learning_rate'],
                "epoch_time": history['epoch_time']
            },
            "per_class_iou_history": {
                f"val_iou_{class_names[i]}": history.get(f'val_iou_class_{i}', [])
                for i in range(config.num_classes)
            },
            "config": asdict(fold_config)
        }
        fold_json_path = os.path.join(str(fold_save_dir), f"fold_{fold_idx}_metrics.json")
        with open(fold_json_path, 'w') as f:
            json.dump(fold_json, f, indent=2, default=str)
        print(f"  JSON saved: {fold_json_path}")
        
        # Stocker pour les stats globales
        all_ious.append(checkpoint_mgr.best_iou)
        all_f1s.append(checkpoint_mgr.best_f1)
        all_losses.append(checkpoint_mgr.best_loss)
        all_accuracies.append(best_metrics['accuracy'])
        
        for i in range(config.num_classes):
            iou_val = best_metrics['per_class_iou'].get(f'iou_class_{i}', 0)
            all_per_class_ious[i].append(iou_val)
            f1_val = best_metrics['per_class_f1'].get(f'f1_class_{i}', 0)
            all_per_class_f1s[i].append(f1_val)
        
        fold_results.append({
            'fold': fold_idx,
            'best_iou': checkpoint_mgr.best_iou,
            'best_f1': checkpoint_mgr.best_f1,
            'best_loss': checkpoint_mgr.best_loss,
            'accuracy': best_metrics['accuracy'],
            'training_epochs': len(history['train_loss']),
            'val_metrics': fold_json['val_metrics'],
            'history': history
        })
        
        print(f"\n✓ Fold {fold_idx + 1} terminé - Best mIoU: {checkpoint_mgr.best_iou:.4f} | "
              f"Best F1: {checkpoint_mgr.best_f1:.4f} | Best Loss: {checkpoint_mgr.best_loss:.4f}")
    
    # ========================================================================
    # STATISTIQUES GLOBALES K-FOLD
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("CALCUL DES STATISTIQUES K-FOLD")
    print(f"{'='*70}")
    
    mean_iou = float(np.mean(all_ious))
    std_iou = float(np.std(all_ious))
    mean_f1 = float(np.mean(all_f1s))
    std_f1 = float(np.std(all_f1s))
    mean_loss = float(np.mean(all_losses))
    std_loss = float(np.std(all_losses))
    mean_acc = float(np.mean(all_accuracies))
    std_acc = float(np.std(all_accuracies))
    
    # Intervalles de confiance 95%
    if n_splits > 1:
        ci_iou = scipy.stats.t.interval(0.95, len(all_ious)-1, loc=mean_iou, scale=scipy.stats.sem(all_ious))
        ci_f1 = scipy.stats.t.interval(0.95, len(all_f1s)-1, loc=mean_f1, scale=scipy.stats.sem(all_f1s))
    else:
        ci_iou = (mean_iou, mean_iou)
        ci_f1 = (mean_f1, mean_f1)
    
    cv_stats = {
        'n_splits': n_splits,
        'mean_iou': mean_iou, 'std_iou': std_iou,
        'mean_f1': mean_f1, 'std_f1': std_f1,
        'mean_loss': mean_loss, 'std_loss': std_loss,
        'mean_accuracy': mean_acc, 'std_accuracy': std_acc,
        'ci_iou_95': {'lower': float(ci_iou[0]), 'upper': float(ci_iou[1])},
        'ci_f1_95': {'lower': float(ci_f1[0]), 'upper': float(ci_f1[1])},
        'all_ious': [float(x) for x in all_ious],
        'all_f1s': [float(x) for x in all_f1s],
        'all_losses': [float(x) for x in all_losses],
        'all_accuracies': [float(x) for x in all_accuracies],
    }
    
    # Statistiques par classe
    cv_stats['per_class'] = {}
    for i in range(config.num_classes):
        c_name = class_names[i]
        if all_per_class_ious[i]:
            cv_stats['per_class'][c_name] = {
                'mean_iou': float(np.mean(all_per_class_ious[i])),
                'std_iou': float(np.std(all_per_class_ious[i])),
                'all_ious': [float(x) for x in all_per_class_ious[i]],
                'mean_f1': float(np.mean(all_per_class_f1s[i])),
                'std_f1': float(np.std(all_per_class_f1s[i])),
                'all_f1s': [float(x) for x in all_per_class_f1s[i]]
            }
    
    # JSON global
    cv_results = {
        'cv_stats': cv_stats,
        'fold_results': [{k: v for k, v in fr.items() if k != 'history'} for fr in fold_results],
        'config': asdict(config),
        'cv_save_dir': str(cv_save_dir)
    }
    
    json_path = cv_save_dir / f"cv_results_{config.model_name.replace('-', '_')}.json"
    with open(json_path, 'w') as f:
        json.dump(cv_results, f, indent=2, default=str)
    
    # Affichage final
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION TERMINÉE")
    print(f"{'='*70}")
    print(f"\nRésultats (n={n_splits} folds):")
    print(f"  mIoU:     {mean_iou:.4f} ± {std_iou:.4f}  [95% CI: {ci_iou[0]:.4f} - {ci_iou[1]:.4f}]")
    print(f"  F1:       {mean_f1:.4f} ± {std_f1:.4f}  [95% CI: {ci_f1[0]:.4f} - {ci_f1[1]:.4f}]")
    print(f"  Loss:     {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    print(f"\nPer-fold results:")
    for fr in fold_results:
        print(f"  Fold {fr['fold']}: mIoU={fr['best_iou']:.4f} | F1={fr['best_f1']:.4f} | "
              f"Loss={fr['best_loss']:.4f} | Epochs={fr['training_epochs']}")
    
    if cv_stats['per_class']:
        print(f"\nPer-class mIoU:")
        for c_name, stats in cv_stats['per_class'].items():
            print(f"  {c_name}: IoU={stats['mean_iou']:.4f} ± {stats['std_iou']:.4f} | "
                  f"F1={stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}")
    
    print(f"\nFichiers sauvegardés:")
    print(f"  Global JSON: {json_path}")
    for fold_idx in range(n_splits):
        fd = cv_save_dir / f"fold_{fold_idx}"
        print(f"  Fold {fold_idx}: {fd}/")
        print(f"    - fold_{fold_idx}_best_iou.pth (checkpoint complet)")
        print(f"    - fold_{fold_idx}_best_loss.pth")
        print(f"    - fold_{fold_idx}_metrics.json")
        print(f"    - fold_{fold_idx}_training_log.csv")
        print(f"    - fold_{fold_idx}_training.png")
    
    return cv_results


# ============================================================================
# CLI - SIMPLIFIÉ
# ============================================================================

def get_available_models():
    return ModelFactory.list_models()

def get_available_losses(mode: str = None):
    if mode == 'binary':
        return sorted(LossFactory.BINARY_LOSSES)
    elif mode == 'multiclass':
        return sorted(LossFactory.MULTICLASS_LOSSES)
    return sorted(LossFactory.BINARY_LOSSES | LossFactory.MULTICLASS_LOSSES)


def main():
    parser = argparse.ArgumentParser(
        description='Modern Architectures Training v3 - Optimisé Géospatial',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
================================================================================
EXEMPLES
================================================================================

# Entraînement basique (augmentation basic)
python {os.path.basename(__file__)} \\
    --model segformer-b2 --dataset_root /data/spartine \\
    --mode binary --classes 2 --in_channels 4 \\
    --aug_level basic

# Augmentation avancée (recommandé)
python {os.path.basename(__file__)} \\
    --model manet --encoder efficientnet-b3 \\
    --dataset_root /data/spartine \\
    --mode binary --loss_type binary_focal_dice \\
    --aug_level advanced --dropout_rate 0.3

# ConvNeXt-Tiny avec UNet (moderne et efficace) ✨
python {os.path.basename(__file__)} \\
    --model unet --encoder convnext_tiny \\
    --dataset_root /data/spartine \\
    --mode binary --aug_level advanced

# ConvNeXt-Small avec UNet (haute performance) ✨✨
python {os.path.basename(__file__)} \\
    --model unet --encoder convnext_small \\
    --dataset_root /data/spartine \\
    --mode multiclass --classes 5 --aug_level aggressive

# ConvNeXt-Base avec UNet (état de l'art) ✨✨✨
python {os.path.basename(__file__)} \\
    --model unet --encoder convnext_base \\
    --dataset_root /data/spartine \\
    --mode binary --aug_level aggressive --epochs 150

# Augmentation aggressive (petit dataset)
python {os.path.basename(__file__)} \\
    --model unet --encoder resnet50 \\
    --dataset_root /data/small \\
    --aug_level aggressive --epochs 200

# Augmentation extreme (très petit dataset)
python {os.path.basename(__file__)} \\
    --model segformer-b3 --dataset_root /data/tiny \\
    --aug_level extreme --dropout_rate 0.5

================================================================================
K-FOLD CROSS-VALIDATION (NOUVEAU!)
================================================================================
# Validation croisée 5-fold (recommandé pour petits datasets)
python {os.path.basename(__file__)} \\
    --model unet --encoder resnet34 \\
    --dataset_root /data/spartine \\
    --mode binary --use_kfold --n_splits 5 \\
    --aug_level advanced

# K-Fold avec ConvNeXt-Tiny + UNet (moderne et robuste) ✨
python {os.path.basename(__file__)} \\
    --model unet --encoder convnext_tiny \\
    --dataset_root /data/spartine \\
    --use_kfold --n_splits 5 --aug_level advanced

# K-Fold avec ConvNeXt-Small + UNet (haute performance) ✨✨
python {os.path.basename(__file__)} \\
    --model unet --encoder convnext_small \\
    --dataset_root /data/spartine \\
    --use_kfold --n_splits 5 --aug_level aggressive

# K-Fold avec 10 folds pour plus de robustesse (autres modèles)
python {os.path.basename(__file__)} \\
    --model manet --encoder efficientnet-b3 \\
    --dataset_root /data/spartine \\
    --use_kfold --n_splits 10 --epochs 50

================================================================================
NIVEAUX D'AUGMENTATION (--aug_level)
================================================================================
  none       : Aucune augmentation
  basic      : Flip H/V + Rotation 90° (invariances géométriques)
  advanced   : + Scale, Brightness, Contrast, Gamma, Channel noise
  aggressive : + Elastic, Blur, Crop, Noise, Dropout, Minority oversample
  extreme    : + MixUp, CutMix, Grid distort, Motion blur, Channel shuffle

================================================================================
ENCODERS RECOMMANDÉS
================================================================================
CLASSIQUES (compatibles avec TOUS les modèles):
  resnet34, resnet50, resnet101
  efficientnet-b0, efficientnet-b3, efficientnet-b4
  se_resnext50_32x4d, senet154

CONVNEXT (⚠️ UNIQUEMENT avec --model unet) ✨:
  convnext_tiny      - Rapide et efficace (28M params)
  convnext_small     - Bon équilibre performance/vitesse (50M params) ⭐
  convnext_base      - Haute performance (89M params) ⭐⭐
  convnext_large     - État de l'art (198M params)
  convnext_xlarge    - Maximum de capacité (350M params)

⚠️ IMPORTANT ConvNeXt:
  ConvNeXt nécessite --model unet (implémentation custom via timm)
  Pour d'autres modèles (MANet, DeepLabV3+, etc.), utilisez:
    - efficientnet-b3 ou efficientnet-b4 (excellent choix)
    - se_resnext50_32x4d (très performant)
    - resnet50 ou resnet101 (classique)

Autres encoders SMP disponibles: senet154, densenet121, mobilenet_v2, etc.
(Voir documentation segmentation_models_pytorch)

================================================================================
LOSSES DISPONIBLES
================================================================================
  Binary: {", ".join(get_available_losses('binary'))}
  Multiclass: {", ".join(get_available_losses('multiclass'))}

================================================================================
MODÈLES DISPONIBLES
================================================================================
  {", ".join(get_available_models())}
================================================================================
        '''
    )
    
    # Core
    core = parser.add_argument_group('Core')
    core.add_argument('--model', type=str, default='segformer-b2', help='Architecture')
    core.add_argument('--dataset_root', type=str, required=True, help='Dataset path')
    core.add_argument('--mode', type=str, default='multiclass', choices=['binary', 'multiclass'])
    core.add_argument('--classes', type=int, default=2, help='Number of classes')
    core.add_argument('--in_channels', type=int, default=4, help='Input channels (1-20+)')
    core.add_argument('--patch_size', type=int, default=224, help='Patch size')
    
    # Model
    model_args = parser.add_argument_group('Model')
    model_args.add_argument('--encoder', type=str, default='resnet34', 
                           help='Encoder backbone (resnet34, efficientnet-b3, convnext_tiny, convnext_small, convnext_base, etc.)')
    model_args.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout (0.0-0.5)')
    model_args.add_argument('--pretrained', action='store_true', default=True)
    model_args.add_argument('--no_pretrained', action='store_true')
    
    # Training
    train_args = parser.add_argument_group('Training')
    train_args.add_argument('--epochs', type=int, default=100)
    train_args.add_argument('--batch_size', type=int, default=8)
    train_args.add_argument('--lr', type=float, default=1e-4)
    train_args.add_argument('--patience', type=int, default=20)
    
    # Loss
    loss_args = parser.add_argument_group('Loss')
    loss_args.add_argument('--loss_type', type=str, default='focal_dice', help='Loss function')
    loss_args.add_argument('--focal_gamma', type=float, default=2.0)
    loss_args.add_argument('--focal_alpha', type=float, default=0.25)
    loss_args.add_argument('--tversky_alpha', type=float, default=0.3)
    loss_args.add_argument('--tversky_beta', type=float, default=0.7)
    loss_args.add_argument('--use_class_weights', action='store_true', default=True)
    loss_args.add_argument('--no_class_weights', action='store_true')
    
    # Optimization
    opt_args = parser.add_argument_group('Optimization')
    opt_args.add_argument('--freeze_encoder', action='store_true')
    opt_args.add_argument('--freeze_epochs', type=int, default=5)
    opt_args.add_argument('--warmup_epochs', type=int, default=0)
    opt_args.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'reduce_plateau', 'one_cycle'])
    opt_args.add_argument('--use_amp', action='store_true', default=True)
    opt_args.add_argument('--no_amp', action='store_true')
    
    # Augmentation - SIMPLIFIÉ
    aug_args = parser.add_argument_group('Data Augmentation')
    aug_args.add_argument('--aug_level', type=str, default='advanced',
                         choices=['none', 'basic', 'advanced', 'aggressive', 'extreme'],
                         help='Niveau d\'augmentation (none/basic/advanced/aggressive/extreme)')
    
    # Output
    out_args = parser.add_argument_group('Output')
    out_args.add_argument('--save_dir', type=str, default='./trained_models')
    out_args.add_argument('--class_names', nargs='+', default=None)
    out_args.add_argument('--device', type=str, default='cuda')
    out_args.add_argument('--num_workers', type=int, default=0)
    
    # K-Fold Cross-Validation
    cv_args = parser.add_argument_group('K-Fold Cross-Validation')
    cv_args.add_argument('--use_kfold', action='store_true', 
                        help='Utiliser la validation croisée K-Fold au lieu du train/val/test classique')
    cv_args.add_argument('--n_splits', type=int, default=5,
                        help='Nombre de folds pour K-Fold CV (défaut: 5)')
    cv_args.add_argument('--cv_subdirs', nargs='+', default=['train', 'validation', 'test'],
                        help='Sous-répertoires à utiliser pour K-Fold (défaut: train validation test)')
    
    # Utility
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--list-losses', action='store_true')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nModèles disponibles:")
        for m in get_available_models():
            print(f"  - {m}")
        return
    
    if args.list_losses:
        print("\nLosses disponibles:")
        print("\n  Mode Binary:")
        for l in get_available_losses('binary'):
            print(f"    - {l}")
        print("\n  Mode Multiclass:")
        for l in get_available_losses('multiclass'):
            print(f"    - {l}")
        return
    
    # Build config
    config = TrainingConfig(
        model_name=args.model,
        encoder_name=args.encoder,
        mode=args.mode,
        num_classes=args.classes,
        in_channels=args.in_channels,
        patch_size=args.patch_size,
        pretrained=not args.no_pretrained,
        dropout_rate=args.dropout_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        use_class_weights=not args.no_class_weights,
        freeze_encoder=args.freeze_encoder,
        freeze_epochs=args.freeze_epochs,
        warmup_epochs=args.warmup_epochs,
        scheduler_type=args.scheduler,
        use_amp=not args.no_amp,
        augmentation_level=args.aug_level,
        num_workers=args.num_workers,
        class_names=args.class_names,
        save_dir=args.save_dir
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Utiliser K-Fold CV ou entraînement classique
    if args.use_kfold:
        print("\n" + "="*70)
        print("MODE: K-FOLD CROSS-VALIDATION")
        print("="*70)
        result = cross_validate_model_v3(
            config=config,
            device=device,
            dataset_root=args.dataset_root,
            n_splits=args.n_splits,
            subdirectories=args.cv_subdirs
        )
        
        print("\n" + "="*70)
        print("CROSS-VALIDATION COMPLETED")
        print("="*70)
        print(f"Results saved to: {result['cv_save_dir']}")
        print(f"Mean mIoU: {result['cv_stats']['mean_iou']:.4f} ± {result['cv_stats']['std_iou']:.4f}")
        print(f"Mean F1: {result['cv_stats']['mean_f1']:.4f} ± {result['cv_stats']['std_f1']:.4f}")
    else:
        print("\n" + "="*70)
        print("MODE: STANDARD TRAINING (train/val/test)")
        print("="*70)
        result = train_model(config, device, args.dataset_root)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Model: {result['model_path']}")
        print(f"Metrics: {result['metrics_path']}")
        print(f"Test mIoU: {result['test_metrics']['mean_iou']:.4f}")


if __name__ == '__main__':
    main()
