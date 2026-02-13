#!/usr/bin/env python3
"""
PREDICT MODERN V3 - Prédicteur d'images satellite/drone grand format
=====================================================================
Compatible avec les modèles entraînés via train_modern_architectures_v3_kfold.py

Architectures supportées:
  - SMP: unet, unet++, deeplabv3+, deeplabv3, manet, fpn, pan, pspnet, linknet
  - SegFormer: segformer-b0 à segformer-b5
  - UNetFormer, HRNet (w18/w32/w48), Swin-UNet

Fonctionnalités:
  - Fenêtre glissante avec chevauchement (overlap)
  - Pondération gaussienne pour fusion sans coutures
  - Normalisation identique à l'entraînement (percentile 99 global)
  - Support binaire (Sigmoid) et multiclasse (Softmax)
  - Sortie GeoTIFF géoréférencée
  - Batch inference GPU
  - Carte de confiance optionnelle

Author: predict_modern_v3
"""

import os
import sys
import math
import argparse
import tempfile
import warnings
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import rasterio
from rasterio.windows import Window

from tqdm import tqdm

warnings.filterwarnings('ignore')


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
# ARCHITECTURES MODERNES (copie exacte du train_modern_architectures_v3)
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

    # Configs architecturales pour chaque variant (pour prédiction sans internet)
    VARIANT_CONFIGS = {
        'segformer-b0': dict(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256],
                             decoder_hidden_size=256, num_attention_heads=[1, 2, 5, 8],
                             mlp_ratios=[4, 4, 4, 4], sr_ratios=[8, 4, 2, 1]),
        'segformer-b1': dict(depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512],
                             decoder_hidden_size=256, num_attention_heads=[1, 2, 5, 8],
                             mlp_ratios=[4, 4, 4, 4], sr_ratios=[8, 4, 2, 1]),
        'segformer-b2': dict(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512],
                             decoder_hidden_size=768, num_attention_heads=[1, 2, 5, 8],
                             mlp_ratios=[4, 4, 4, 4], sr_ratios=[8, 4, 2, 1]),
        'segformer-b3': dict(depths=[3, 4, 18, 3], hidden_sizes=[64, 128, 320, 512],
                             decoder_hidden_size=768, num_attention_heads=[1, 2, 5, 8],
                             mlp_ratios=[4, 4, 4, 4], sr_ratios=[8, 4, 2, 1]),
        'segformer-b4': dict(depths=[3, 8, 27, 3], hidden_sizes=[64, 128, 320, 512],
                             decoder_hidden_size=768, num_attention_heads=[1, 2, 5, 8],
                             mlp_ratios=[4, 4, 4, 4], sr_ratios=[8, 4, 2, 1]),
        'segformer-b5': dict(depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512],
                             decoder_hidden_size=768, num_attention_heads=[1, 2, 5, 8],
                             mlp_ratios=[4, 4, 4, 4], sr_ratios=[8, 4, 2, 1]),
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
            # Entraînement : charger poids pré-entraînés + fine-tune
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.VARIANTS[variant],
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                hidden_dropout_prob=dropout_rate,
                attention_probs_dropout_prob=dropout_rate
            )
        elif variant in self.VARIANTS:
            # Prédiction (pretrained=False) : construire la bonne architecture
            # Méthode 1 : charger la config depuis HuggingFace (cache local si dispo)
            config = None
            try:
                config = SegformerConfig.from_pretrained(
                    self.VARIANTS[variant],
                    num_labels=num_classes,
                    num_channels=in_channels,
                    hidden_dropout_prob=dropout_rate,
                    attention_probs_dropout_prob=dropout_rate
                )
                print(f"    ✓ Config SegFormer '{variant}' chargée depuis cache/HuggingFace")
            except Exception:
                pass

            # Méthode 2 : config hardcodée (fallback offline)
            if config is None and variant in self.VARIANT_CONFIGS:
                variant_params = self.VARIANT_CONFIGS[variant]
                config = SegformerConfig(
                    num_labels=num_classes,
                    num_channels=in_channels,
                    hidden_dropout_prob=dropout_rate,
                    attention_probs_dropout_prob=dropout_rate,
                    **variant_params
                )
                print(f"    ✓ Config SegFormer '{variant}' chargée depuis table locale")

            if config is None:
                raise ValueError(f"Impossible de charger la config pour '{variant}'")

            self.model = SegformerForSemanticSegmentation(config)
        else:
            # Fallback : config par défaut (B0)
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
            skip_ch = self.encoder_channels[-(i + 2)] if i < len(self.encoder_channels) - 1 else 0
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
                skip = features[-(i + 2)]
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

    def __init__(self, num_classes: int, in_channels: int, pretrained: bool = True,
                 dropout_rate: float = 0.3):
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
# MODEL FACTORY (parité exacte avec train_modern_architectures_v3_kfold.py)
# ============================================================================

class ModelFactory:
    """Factory pour instancier tous les modèles supportés par le système v3"""

    MODERN_MODELS = {
        'segformer-b0': ('segformer', 'b0'), 'segformer-b1': ('segformer', 'b1'),
        'segformer-b2': ('segformer', 'b2'), 'segformer-b3': ('segformer', 'b3'),
        'segformer-b4': ('segformer', 'b4'), 'segformer-b5': ('segformer', 'b5'),
        'unetformer': ('unetformer', None),
        'hrnet-w18': ('hrnet', 'hrnet_w18'), 'hrnet-w32': ('hrnet', 'hrnet_w32'),
        'hrnet-w48': ('hrnet', 'hrnet_w48'),
        'swin-unet': ('swin-unet', None),
    }

    SMP_MODELS = ['unet', 'unet++', 'deeplabv3+', 'deeplabv3', 'manet', 'fpn', 'pan', 'pspnet', 'linknet']

    @classmethod
    def list_models(cls):
        models = list(cls.MODERN_MODELS.keys())
        if DEPS.get('smp'):
            models.extend(cls.SMP_MODELS)
        return models

    @classmethod
    def build_model(cls, model_name: str, encoder_name: str = 'resnet34',
                    in_channels: int = 4, num_classes: int = 2,
                    mode: str = 'multiclass', pretrained: bool = False,
                    dropout_rate: float = 0.3) -> nn.Module:
        """
        Construit un modèle de segmentation.
        Parité exacte avec ModelFactory.create() du script d'entraînement.

        Args:
            model_name: Nom de l'architecture (ex: 'unet', 'segformer-b2', etc.)
            encoder_name: Backbone encoder (pour SMP et UNetFormer)
            in_channels: Nombre de canaux d'entrée
            num_classes: Nombre de classes total
            mode: 'binary' ou 'multiclass'
            pretrained: Charger les poids pré-entraînés (False pour prédiction)
            dropout_rate: Taux de dropout
        """
        name = model_name.lower()
        # En binary, le modèle a 1 sortie ; en multiclass, num_classes sorties
        actual_classes = 1 if mode == 'binary' else num_classes

        # --- Architectures modernes (SegFormer, UNetFormer, HRNet, Swin-UNet) ---
        if name in cls.MODERN_MODELS:
            model_type, variant = cls.MODERN_MODELS[name]

            if model_type == 'segformer':
                return SegFormerWrapper(name, actual_classes, in_channels, pretrained, dropout_rate)
            elif model_type == 'unetformer':
                return UNetFormer(actual_classes, in_channels, encoder_name, pretrained, dropout_rate)
            elif model_type == 'hrnet':
                return HRNetSegmentation(variant, actual_classes, in_channels, pretrained, dropout_rate)
            elif model_type == 'swin-unet':
                return SwinUNet(actual_classes, in_channels, pretrained, dropout_rate)

        # --- Modèles SMP ---
        elif DEPS.get('smp'):
            import segmentation_models_pytorch as smp
            smp_map = {
                'unet': smp.Unet, 'unet++': smp.UnetPlusPlus,
                'deeplabv3+': smp.DeepLabV3Plus, 'deeplabv3': smp.DeepLabV3,
                'manet': smp.MAnet, 'fpn': smp.FPN,
                'pan': smp.PAN, 'pspnet': smp.PSPNet, 'linknet': smp.Linknet,
            }
            # Normalisation du nom pour matching
            name_clean = name.replace('-', '').replace('_', '')
            for key, model_cls in smp_map.items():
                key_clean = key.replace('+', 'plus').replace('-', '').replace('_', '')
                # Aussi tester avec le '+' tel quel
                if name_clean == key_clean or name == key:
                    return model_cls(
                        encoder_name=encoder_name,
                        in_channels=in_channels,
                        classes=actual_classes,
                        encoder_weights='imagenet' if pretrained else None,
                        activation=None
                    )

        raise ValueError(
            f"Modèle inconnu: '{model_name}'. Disponibles: {cls.list_models()}"
        )


# ============================================================================
# PRÉDICTEUR D'IMAGES GRAND FORMAT
# ============================================================================

class LargeImagePredictor:
    """
    Prédicteur d'images satellite/drone grand format avec reconstruction seamless.

    Compatible avec les modèles entraînés via train_modern_architectures_v3_kfold.py.
    Utilise une fenêtre glissante avec pondération gaussienne pour éliminer
    les artefacts de bordure.
    """

    def __init__(self, model_path: str, model_name: str, encoder_name: str = 'resnet34',
                 in_channels: int = 4, num_classes: int = 2,
                 patch_size: int = 224, overlap: int = 112,
                 batch_size: int = 4, threshold: float = 0.5,
                 device: str = 'cuda', dropout_rate: float = 0.3):
        """
        Args:
            model_path: Chemin vers le fichier .pth
            model_name: Architecture (unet, segformer-b2, deeplabv3+, etc.)
            encoder_name: Backbone encoder (resnet34, efficientnet-b3, etc.)
            in_channels: Nombre de canaux d'entrée
            num_classes: Nombre de classes (1 = binary, >1 = multiclass)
            patch_size: Taille des patches (doit correspondre à l'entraînement)
            overlap: Chevauchement entre patches en pixels
            batch_size: Taille de batch pour l'inférence GPU
            threshold: Seuil pour la segmentation binaire
            device: 'cuda' ou 'cpu'
            dropout_rate: Taux de dropout (doit correspondre à l'entraînement)
        """
        self.model_path = model_path
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout_rate = dropout_rate
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

        # Déterminer le mode
        self.mode = 'binary' if num_classes == 1 else 'multiclass'

        self._print_config()

        # Charger le modèle
        self.model = self._load_model()

        # Pré-calculer la fenêtre de pondération gaussienne
        self.gaussian_window = self._create_gaussian_window(patch_size, patch_size)

    def _print_config(self):
        """Affiche la configuration du prédicteur"""
        print(f"\n{'=' * 70}")
        print("PREDICT MODERN V3 - INITIALISATION")
        print(f"{'=' * 70}")
        print(f"  Modèle         : {self.model_name}")
        print(f"  Encoder         : {self.encoder_name}")
        print(f"  Poids           : {self.model_path}")
        print(f"  Device          : {self.device}")
        print(f"  Canaux entrée   : {self.in_channels}")
        print(f"  Classes         : {self.num_classes}")
        print(f"  Mode            : {self.mode.upper()}")
        print(f"  Patch size      : {self.patch_size}")
        print(f"  Overlap         : {self.overlap}")
        print(f"  Batch size      : {self.batch_size}")
        if self.mode == 'binary':
            print(f"  Seuil           : {self.threshold}")
        print(f"  Dropout rate    : {self.dropout_rate}")

    # ------------------------------------------------------------------
    # CHARGEMENT DU MODÈLE
    # ------------------------------------------------------------------

    def _load_model(self) -> nn.Module:
        """
        Charge le modèle avec détection automatique du format de checkpoint.
        Gère les formats:
          - state_dict pur (K-Fold)
          - {'model_state_dict': ..., 'config': ...} (standard v3)
          - {'state_dict': ...}
        """
        print(f"\nChargement du modèle depuis: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Fichier modèle introuvable: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)

        # --- Extraire le state_dict et la config ---
        state_dict = None
        config = {}

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                config = checkpoint.get('config', {})
                print("  ✓ Format détecté: checkpoint complet v3 (model_state_dict + config)")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                config = checkpoint.get('config', checkpoint.get('metadata', {}))
                print("  ✓ Format détecté: state_dict wrapper")
            else:
                # Vérifier si c'est un state_dict pur (les clés ressemblent à des paramètres)
                sample_keys = list(checkpoint.keys())[:5]
                looks_like_state_dict = any(
                    '.' in k and any(kw in k for kw in ['weight', 'bias', 'running_mean', 'num_batches'])
                    for k in sample_keys
                )
                if looks_like_state_dict:
                    state_dict = checkpoint
                    print("  ✓ Format détecté: state_dict pur (K-Fold)")
                else:
                    state_dict = checkpoint
                    print("  ⚠ Format inconnu - tentative de chargement direct")
        else:
            raise ValueError(f"Type de checkpoint inattendu: {type(checkpoint)}")

        # --- Mise à jour des paramètres depuis la config sauvegardée ---
        if config:
            self._update_params_from_config(config)

        # --- Construire le modèle ---
        print(f"\nConstruction de l'architecture: {self.model_name}")
        model = ModelFactory.build_model(
            model_name=self.model_name,
            encoder_name=self.encoder_name,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            mode=self.mode,
            pretrained=False,
            dropout_rate=self.dropout_rate
        )

        # --- Charger les poids ---
        # Retirer le préfixe 'module.' si DataParallel
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        try:
            model.load_state_dict(state_dict, strict=True)
            print("  ✓ Poids chargés avec succès (strict=True)")
        except RuntimeError as e:
            print(f"  ⚠ Chargement strict échoué: {e}")
            print("  → Tentative avec strict=False...")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"    Clés manquantes ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"    Clés inattendues ({len(unexpected)}): {unexpected[:5]}...")
            print("  ✓ Poids chargés avec strict=False")

        model.to(self.device)
        model.eval()
        print(f"  ✓ Modèle en mode eval sur {self.device}")

        # Compter les paramètres
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Paramètres totaux: {total_params:,}")

        return model

    def _update_params_from_config(self, config: dict):
        """Met à jour les paramètres depuis la config sauvegardée"""
        detected = {}
        for key in ['model_name', 'encoder_name', 'in_channels', 'num_classes', 'mode',
                     'patch_size', 'dropout_rate']:
            if key in config:
                detected[key] = config[key]

        if detected:
            print(f"\n  Config détectée dans le checkpoint:")
            for k, v in detected.items():
                print(f"    {k}: {v}")

            # === Auto-correction du dropout_rate depuis le checkpoint ===
            if 'dropout_rate' in detected and detected['dropout_rate'] != self.dropout_rate:
                old_dr = self.dropout_rate
                self.dropout_rate = detected['dropout_rate']
                print(f"  → dropout_rate auto-corrigé: {old_dr} → {self.dropout_rate} (depuis checkpoint)")

            # === Auto-correction num_classes pour le mode binary ===
            # Dans le training v3, config sauvegarde num_classes=2 (le paramètre CLI)
            # mais le modèle réel a 1 sortie quand mode=binary.
            # On détecte et on corrige automatiquement.
            detected_mode = detected.get('mode', None)
            if detected_mode == 'binary' and self.num_classes != 1:
                print(f"  → Mode binary détecté: num_classes forcé à 1 (sortie Sigmoid)")
                self.num_classes = 1
                self.mode = 'binary'
            elif detected_mode == 'binary' and self.num_classes == 1:
                self.mode = 'binary'

            # Avertissements si décalage (les args CLI priment sauf pour les auto-corrections)
            if 'model_name' in detected and detected['model_name'] != self.model_name:
                print(f"  ⚠ Attention: model_name checkpoint='{detected['model_name']}' vs CLI='{self.model_name}'")
            if 'in_channels' in detected and detected['in_channels'] != self.in_channels:
                print(f"  ⚠ Attention: in_channels checkpoint={detected['in_channels']} vs CLI={self.in_channels}")
            if 'encoder_name' in detected and detected['encoder_name'] != self.encoder_name:
                print(f"  ⚠ Note: encoder_name checkpoint='{detected['encoder_name']}' vs CLI='{self.encoder_name}'")

    # ------------------------------------------------------------------
    # NORMALISATION (parité exacte avec l'entraînement)
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_patch(patch: np.ndarray) -> np.ndarray:
        """
        Normalisation identique à SegmentationDataset.__getitem__() du script
        d'entraînement train_modern_architectures_v3_kfold.py.

        Méthode: percentile 99 global sur le patch, puis clip [0, 1].
        C'est une normalisation GLOBALE (pas par canal) pour rester en parité.

        Args:
            patch: (C, H, W) en float32

        Returns:
            patch normalisé (C, H, W) dans [0, 1]
        """
        patch = patch.astype(np.float32)
        if patch.max() > 0:
            p99 = np.percentile(patch, 99)
            patch = np.clip(patch / (p99 + 1e-6), 0, 1)
        return patch

    # ------------------------------------------------------------------
    # FENÊTRE DE PONDÉRATION GAUSSIENNE
    # ------------------------------------------------------------------

    @staticmethod
    def _create_gaussian_window(height: int, width: int, sigma_scale: float = 0.25) -> np.ndarray:
        """
        Crée une fenêtre de pondération gaussienne 2D pour la fusion seamless.

        Le centre du patch a le poids maximal (1.0), les bords ont un poids
        faible. Cela élimine les artefacts de grille dans les zones de
        recouvrement.

        Args:
            height: Hauteur du patch
            width: Largeur du patch
            sigma_scale: Proportion de la taille pour sigma (0.25 = sigma = size/4)

        Returns:
            Fenêtre gaussienne (H, W)
        """
        sigma_y = height * sigma_scale
        sigma_x = width * sigma_scale

        y = np.arange(height) - (height - 1) / 2.0
        x = np.arange(width) - (width - 1) / 2.0

        gy = np.exp(-(y ** 2) / (2 * sigma_y ** 2))
        gx = np.exp(-(x ** 2) / (2 * sigma_x ** 2))

        window = np.outer(gy, gx).astype(np.float32)

        # Normaliser pour que le max = 1.0
        window /= window.max()

        # Plancher minimal pour éviter les zones à poids nul
        window = np.clip(window, 0.01, 1.0)

        return window

    # ------------------------------------------------------------------
    # EXTRACTION DES PATCHES
    # ------------------------------------------------------------------

    def _extract_patch_grid(self, img_height: int, img_width: int):
        """
        Calcule les positions (x, y) de la grille de patches avec overlap.

        Returns:
            Liste de tuples (x, y) correspondant au coin supérieur-gauche
        """
        step = self.patch_size - self.overlap
        step = max(step, 1)

        positions = []

        y = 0
        while y < img_height:
            x = 0
            while x < img_width:
                # Ajustement pour ne pas dépasser
                actual_x = min(x, max(0, img_width - self.patch_size))
                actual_y = min(y, max(0, img_height - self.patch_size))
                positions.append((actual_x, actual_y))
                x += step
                if x >= img_width and (x - step + self.patch_size) < img_width:
                    # Ajouter un dernier patch calé à droite
                    positions.append((max(0, img_width - self.patch_size), actual_y))
                    break
            y += step
            if y >= img_height and (y - step + self.patch_size) < img_height:
                # Ajouter une dernière rangée calée en bas
                x = 0
                while x < img_width:
                    actual_x = min(x, max(0, img_width - self.patch_size))
                    positions.append((actual_x, max(0, img_height - self.patch_size)))
                    x += step
                    if x >= img_width and (x - step + self.patch_size) < img_width:
                        positions.append((max(0, img_width - self.patch_size),
                                         max(0, img_height - self.patch_size)))
                        break
                break

        # Dédupliquer
        positions = list(dict.fromkeys(positions))
        return positions

    # ------------------------------------------------------------------
    # INFÉRENCE PAR BATCH
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_batch(self, batch_tensor: torch.Tensor) -> np.ndarray:
        """
        Inférence sur un batch de patches.

        Args:
            batch_tensor: (B, C, H, W) tensor sur le device

        Returns:
            Probabilités numpy:
              - Binary: (B, H, W)
              - Multiclass: (B, num_classes, H, W)
        """
        output = self.model(batch_tensor)

        # Gérer différents formats de sortie
        if isinstance(output, dict):
            output = output.get('out', output.get('logits', list(output.values())[0]))
        elif isinstance(output, (tuple, list)):
            output = output[0]

        if self.mode == 'binary':
            probs = torch.sigmoid(output).squeeze(1)  # (B, H, W)
        else:
            probs = torch.softmax(output, dim=1)  # (B, C, H, W)

        return probs.cpu().numpy()

    # ------------------------------------------------------------------
    # PRÉDICTION PRINCIPALE
    # ------------------------------------------------------------------

    def predict(self, input_path: str, output_path: str,
                save_confidence: bool = False, output_nodata: int = 255):
        """
        Prédit une image grand format avec reconstruction seamless.

        Args:
            input_path: Chemin vers le GeoTIFF d'entrée
            output_path: Chemin de sortie pour le masque GeoTIFF
            save_confidence: Sauvegarder la carte de confiance
            output_nodata: Valeur NoData pour la sortie
        """
        print(f"\n{'=' * 70}")
        print("DÉMARRAGE DE LA PRÉDICTION")
        print(f"{'=' * 70}")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Image d'entrée introuvable: {input_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # ====== ÉTAPE 1: Lire les métadonnées ======
        print(f"\n[1/4] Lecture des métadonnées...")
        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            img_height = src.height
            img_width = src.width
            img_channels = src.count
            nodata_value = profile.get('nodata', None)
            crs = src.crs
            transform = src.transform

        print(f"  Taille   : {img_width} x {img_height} pixels")
        print(f"  Canaux   : {img_channels}")
        print(f"  CRS      : {crs}")
        print(f"  NoData   : {nodata_value}")

        if img_channels != self.in_channels:
            print(f"  ⚠ L'image a {img_channels} canaux, le modèle attend {self.in_channels}")
            if img_channels < self.in_channels:
                print(f"    → Les canaux seront répétés pour correspondre")

        # Gestion des images plus petites que le patch
        if img_height < self.patch_size or img_width < self.patch_size:
            print(f"  ⚠ Image plus petite que le patch_size ({self.patch_size})")
            print(f"    → L'image sera paddée")

        # ====== ÉTAPE 2: Calculer la grille ======
        print(f"\n[2/4] Calcul de la grille de patches...")

        # Padder si nécessaire
        pad_h = max(0, self.patch_size - img_height)
        pad_w = max(0, self.patch_size - img_width)
        effective_h = img_height + pad_h
        effective_w = img_width + pad_w

        positions = self._extract_patch_grid(effective_h, effective_w)
        n_patches = len(positions)

        step = self.patch_size - self.overlap
        print(f"  Grille   : {n_patches} patches")
        print(f"  Step     : {step} pixels")
        print(f"  Overlap  : {self.overlap} pixels")

        # ====== ÉTAPE 3: Prédiction par batch ======
        print(f"\n[3/4] Prédiction des patches (batch_size={self.batch_size})...")

        # Accumulateurs pour la reconstruction
        if self.mode == 'binary':
            weighted_sum = np.zeros((effective_h, effective_w), dtype=np.float64)
        else:
            weighted_sum = np.zeros((self.num_classes, effective_h, effective_w), dtype=np.float64)
        weight_sum = np.zeros((effective_h, effective_w), dtype=np.float64)

        # Traitement par batch
        with rasterio.open(input_path) as src:
            batch_patches = []
            batch_positions = []

            for pos_idx in tqdm(range(n_patches), desc="Prédiction", unit="patch"):
                x, y = positions[pos_idx]

                # Lire le patch depuis le raster (en tenant compte du padding)
                read_x = min(x, img_width - 1)
                read_y = min(y, img_height - 1)
                read_w = min(self.patch_size, img_width - read_x)
                read_h = min(self.patch_size, img_height - read_y)

                window = Window(read_x, read_y, read_w, read_h)
                patch = src.read(window=window)  # (C, H, W)

                # Padder si le patch est plus petit que patch_size
                if patch.shape[1] < self.patch_size or patch.shape[2] < self.patch_size:
                    padded = np.zeros((patch.shape[0], self.patch_size, self.patch_size),
                                     dtype=patch.dtype)
                    padded[:, :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded

                # Adapter le nombre de canaux
                if patch.shape[0] < self.in_channels:
                    repeats = math.ceil(self.in_channels / patch.shape[0])
                    patch = np.repeat(patch, repeats, axis=0)[:self.in_channels]
                elif patch.shape[0] > self.in_channels:
                    patch = patch[:self.in_channels]

                # Normalisation identique à l'entraînement
                patch_norm = self.normalize_patch(patch)

                batch_patches.append(patch_norm)
                batch_positions.append((x, y))

                # Quand le batch est plein ou c'est le dernier patch
                if len(batch_patches) == self.batch_size or pos_idx == n_patches - 1:
                    batch_tensor = torch.from_numpy(np.stack(batch_patches)).float().to(self.device)
                    batch_probs = self._predict_batch(batch_tensor)

                    # Accumuler chaque patch du batch
                    for b_idx in range(len(batch_patches)):
                        bx, by = batch_positions[b_idx]
                        probs = batch_probs[b_idx]

                        # Déterminer la taille effective (sans padding)
                        eff_h = min(self.patch_size, effective_h - by)
                        eff_w = min(self.patch_size, effective_w - bx)

                        # Fenêtre gaussienne (tronquée si nécessaire)
                        gw = self.gaussian_window[:eff_h, :eff_w]

                        if self.mode == 'binary':
                            weighted_sum[by:by + eff_h, bx:bx + eff_w] += probs[:eff_h, :eff_w] * gw
                        else:
                            for c in range(self.num_classes):
                                weighted_sum[c, by:by + eff_h, bx:bx + eff_w] += probs[c, :eff_h, :eff_w] * gw

                        weight_sum[by:by + eff_h, bx:bx + eff_w] += gw

                    batch_patches.clear()
                    batch_positions.clear()

        # ====== ÉTAPE 4: Reconstruction et sauvegarde ======
        print(f"\n[4/4] Reconstruction et sauvegarde...")

        # Éviter la division par zéro
        weight_sum = np.maximum(weight_sum, 1e-8)

        if self.mode == 'binary':
            final_probs = weighted_sum / weight_sum
            # Tronquer au format original (retirer le padding)
            final_probs = final_probs[:img_height, :img_width]
            final_mask = (final_probs > self.threshold).astype(np.uint8)
            confidence = np.where(final_probs > self.threshold, final_probs, 1 - final_probs)
        else:
            for c in range(self.num_classes):
                weighted_sum[c] /= weight_sum
            weighted_sum = weighted_sum[:, :img_height, :img_width]
            final_mask = np.argmax(weighted_sum, axis=0).astype(np.uint8)
            confidence = np.max(weighted_sum, axis=0).astype(np.float32)

        # Appliquer le masque NoData d'origine
        if nodata_value is not None:
            print("  Application du masque NoData source...")
            with rasterio.open(input_path) as src:
                first_band = src.read(1)
                nodata_mask = (first_band == nodata_value)
                if np.any(nodata_mask):
                    final_mask[nodata_mask] = output_nodata
                    confidence[nodata_mask] = 0

        # Sauvegarder le masque
        output_profile = profile.copy()
        output_profile.update({
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw',
            'nodata': output_nodata
        })

        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(final_mask, 1)
        print(f"  ✓ Masque sauvegardé: {output_path}")

        # Sauvegarder la carte de confiance
        if save_confidence:
            base_name = os.path.splitext(output_path)[0]
            conf_path = f"{base_name}_confidence.tif"
            conf_profile = output_profile.copy()
            conf_profile.update({
                'dtype': 'float32',
                'nodata': -9999.0
            })
            with rasterio.open(conf_path, 'w', **conf_profile) as dst:
                dst.write(confidence.astype(np.float32), 1)
            print(f"  ✓ Carte de confiance: {conf_path}")

        # Statistiques
        self._print_statistics(final_mask, confidence, output_nodata, transform)

        print(f"\n{'=' * 70}")
        print("PRÉDICTION TERMINÉE")
        print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # STATISTIQUES
    # ------------------------------------------------------------------

    def _print_statistics(self, mask: np.ndarray, confidence: np.ndarray,
                          output_nodata: int, geo_transform):
        """Affiche les statistiques de la prédiction"""
        valid_mask = mask != output_nodata
        if not np.any(valid_mask):
            print("  Aucun pixel valide dans la prédiction!")
            return

        valid_pixels = mask[valid_mask]
        unique, counts = np.unique(valid_pixels, return_counts=True)
        total = counts.sum()

        print(f"\n  Statistiques de prédiction ({self.mode}):")
        for cls, count in zip(unique, counts):
            pct = (count / total) * 100
            if self.mode == 'binary':
                label = "Avant-plan" if cls == 1 else "Arrière-plan"
                print(f"    {label}: {count:,} pixels ({pct:.2f}%)")
            else:
                print(f"    Classe {cls:2d}: {count:10,} pixels ({pct:.2f}%)")

        if confidence is not None:
            valid_conf = confidence[valid_mask]
            if len(valid_conf) > 0:
                print(f"\n  Confiance:")
                print(f"    Moyenne : {np.mean(valid_conf):.4f}")
                print(f"    Min/Max : {np.min(valid_conf):.4f} / {np.max(valid_conf):.4f}")
                print(f"    Haute (>0.8): {np.sum(valid_conf > 0.8):,} pixels "
                      f"({np.sum(valid_conf > 0.8) / len(valid_conf) * 100:.1f}%)")

        # Surface si géoréférencé
        if geo_transform:
            try:
                pixel_area = abs(geo_transform[0] * geo_transform[4])
                if pixel_area > 0:
                    total_area = total * pixel_area
                    print(f"\n  Surface totale: {total_area:,.2f} unités²")
                    print(f"  Résolution pixel: {abs(geo_transform[0]):.4f} x {abs(geo_transform[4]):.4f}")
            except (TypeError, IndexError):
                pass


# ============================================================================
# CLI - INTERFACE EN LIGNE DE COMMANDE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prédicteur d\'images grand format pour Modern Architectures v3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
================================================================================
EXEMPLES D'UTILISATION
================================================================================

# Segmentation binaire avec UNet + ResNet34
python predict_modern_v3.py \\
    --model_path best_model.pth --input ortho.tif --output prediction.tif \\
    --model_name unet --encoder_name resnet34 \\
    --in_channels 4 --num_classes 1 --patch_size 224

# Multiclasse avec SegFormer-B2
python predict_modern_v3.py \\
    --model_path segformer_best.pth --input satellite.tif --output classes.tif \\
    --model_name segformer-b2 --in_channels 4 --num_classes 5 --patch_size 224

# Multiclasse avec DeepLabV3+ et EfficientNet
python predict_modern_v3.py \\
    --model_path deeplabv3plus_best.pth --input drone.tif --output seg.tif \\
    --model_name deeplabv3+ --encoder_name efficientnet-b3 \\
    --in_channels 3 --num_classes 6 --patch_size 512 --overlap 128

# Avec carte de confiance et batch GPU
python predict_modern_v3.py \\
    --model_path model.pth --input image.tif --output result.tif \\
    --model_name manet --encoder_name resnet50 \\
    --in_channels 4 --num_classes 2 --batch_size 8 --save_confidence

# K-Fold model (state_dict pur)
python predict_modern_v3.py \\
    --model_path best_model_fold_0.pth --input image.tif --output pred.tif \\
    --model_name unet --encoder_name resnet34 \\
    --in_channels 4 --num_classes 1 --patch_size 224

================================================================================
MODÈLES DISPONIBLES
================================================================================
  SMP        : unet, unet++, deeplabv3+, deeplabv3, manet, fpn, pan, pspnet, linknet
  SegFormer  : segformer-b0, segformer-b1, segformer-b2, segformer-b3, segformer-b4, segformer-b5
  Autres     : unetformer, hrnet-w18, hrnet-w32, hrnet-w48, swin-unet
================================================================================
        '''
    )

    # Arguments requis
    req = parser.add_argument_group('Arguments requis')
    req.add_argument('--model_path', required=True, help='Chemin vers le fichier .pth')
    req.add_argument('--input', required=True, help='Chemin vers l\'image GeoTIFF d\'entrée')
    req.add_argument('--output', required=True, help='Chemin de sortie pour le masque GeoTIFF')
    req.add_argument('--model_name', required=True,
                     help='Architecture du modèle (unet, segformer-b2, deeplabv3+, etc.)')
    req.add_argument('--encoder_name', default='resnet34',
                     help='Backbone encoder (resnet34, efficientnet-b3, etc.) [défaut: resnet34]')
    req.add_argument('--in_channels', type=int, required=True,
                     help='Nombre de canaux d\'entrée (ex: 3 pour RGB, 4 pour RGBN)')
    req.add_argument('--num_classes', type=int, required=True,
                     help='Nombre de classes (1 = binaire, N = multiclasse)')

    # Arguments optionnels
    opt = parser.add_argument_group('Arguments optionnels')
    opt.add_argument('--patch_size', type=int, default=224,
                     help='Taille des patches en pixels [défaut: 224]')
    opt.add_argument('--overlap', type=int, default=112,
                     help='Chevauchement entre patches en pixels [défaut: 112]')
    opt.add_argument('--batch_size', type=int, default=4,
                     help='Taille de batch pour l\'inférence GPU [défaut: 4]')
    opt.add_argument('--threshold', type=float, default=0.5,
                     help='Seuil pour segmentation binaire [défaut: 0.5]')
    opt.add_argument('--dropout_rate', type=float, default=0.3,
                     help='Taux de dropout du modèle [défaut: 0.3]')
    opt.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                     help='Device d\'inférence [défaut: cuda]')
    opt.add_argument('--output_nodata', type=int, default=255,
                     help='Valeur NoData pour la sortie [défaut: 255]')
    opt.add_argument('--save_confidence', action='store_true',
                     help='Sauvegarder la carte de confiance')

    # Utilitaire
    parser.add_argument('--list_models', action='store_true',
                        help='Lister les modèles disponibles et quitter')

    args = parser.parse_args()

    # Lister les modèles
    if args.list_models:
        print("\nModèles disponibles:")
        for m in ModelFactory.list_models():
            print(f"  - {m}")
        return

    # Validations
    if not os.path.exists(args.model_path):
        print(f"✗ Erreur: Fichier modèle introuvable: {args.model_path}")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"✗ Erreur: Image d'entrée introuvable: {args.input}")
        sys.exit(1)

    # Lancer la prédiction
    try:
        predictor = LargeImagePredictor(
            model_path=args.model_path,
            model_name=args.model_name,
            encoder_name=args.encoder_name,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            threshold=args.threshold,
            device=args.device,
            dropout_rate=args.dropout_rate
        )

        predictor.predict(
            input_path=args.input,
            output_path=args.output,
            save_confidence=args.save_confidence,
            output_nodata=args.output_nodata
        )

    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
