# CHANGELOG - SemanticSeg4EO V2

## Version 2.0.0 - Major Update

This major version introduces numerous improvements for more performant and flexible training.

---

## 📦 Modified Files

### 1. `model_training.py` → `model_training_v2.py`

#### New Features

| Feature | Description |
|---------|-------------|
| **TrainingConfig dataclass** | Centralized configuration with automatic validation |
| **Focal Loss** | Class imbalance handling with alpha/gamma parameters |
| **Dice Loss** | Pure Dice loss function |
| **Tversky Loss** | FP/FN tradeoff control with alpha/beta |
| **Combo Loss** | Combined CE + Dice + Focal |
| **Encoder Freezing** | Freeze encoder for transfer learning |
| **LR Warmup** | Gradual learning rate increase |
| **Mixed Precision (AMP)** | Faster training with less memory |
| **Per-class IoU logging** | Detailed metrics per class |
| **CSV Export** | Export metrics to CSV format |
| **New schedulers** | Cosine Annealing, One-Cycle Policy |

#### Technical Details

```python
# New configuration dataclass
@dataclass
class TrainingConfig:
    mode: str = 'multiclass'
    num_classes: int = 5
    loss_type: str = 'dice_ce'  # 'ce', 'dice', 'focal', 'focal_dice', 'tversky', 'combo'
    freeze_encoder: bool = False
    freeze_epochs: int = 5
    warmup_epochs: int = 0
    use_amp: bool = False
    log_per_class_metrics: bool = True
    save_csv_logs: bool = True
    # ... and more
```

#### New Loss Classes

```python
class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, weight=None)

class DiceLoss(nn.Module):
    """Pure Dice Loss"""
    def __init__(self, smooth=1e-6, multiclass=True)

class TverskyLoss(nn.Module):
    """Tversky Loss - Dice generalization"""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6)
```

#### New Training Functions

- `train_and_save_model_v2()` - Simplified version
- `train_and_save_model_v2_full()` - Full version with all options
- `cross_validate_model_v2()` - CV with new features support
- `get_available_losses()` - List available loss functions

---

### 2. `main.py` → `main_v2.py`

#### Main Changes

| Old | New |
|-----|-----|
| Flat arguments | Organized argument groups |
| Fixed loss (Dice+BCE) | 7 configurable loss types |
| No encoder freezing | `--freeze_encoder` support |
| No warmup | `--warmup_epochs` support |
| No AMP | `--use_amp` support |
| No class names | `--class_names` support |

#### New Argument Groups

```bash
# Core Settings
--mode, --dataset_root, --model, --classes, --val_strategy

# Data Configuration
--patch_subdir, --in_channels, --patch_size, --data_augmentation

# Training Configuration  
--epochs, --batch_size, --learning_rate, --patience

# Model Configuration
--encoder_name, --pretrained, --dropout_rate

# Loss Configuration (NEW)
--loss_type, --use_class_weights, --focal_gamma, --focal_alpha,
--tversky_alpha, --tversky_beta, --dice_weight

# Encoder Freezing (NEW)
--freeze_encoder, --freeze_epochs

# Learning Rate Schedule (NEW)
--scheduler_type, --warmup_epochs, --warmup_lr

# Mixed Precision Training (NEW)
--use_amp

# Logging (NEW)
--log_per_class, --class_names, --save_csv
```

---

### 3. `Patch_extraction.py` → `Patch_extraction_v2.py`

#### New Features

| Feature | Description |
|---------|-------------|
| **Batch mode** | Process multiple image-label pairs |
| **Pattern matching** | Automatic detection with regex |
| **ImageLabelPair dataclass** | Structure for pairs |
| **`info` command** | Display dataset information |
| **Per-image grids** | Support Grid_X.shp per Image_X.tif |
| **Progress bar** | tqdm integration |

#### New CLI Commands

```bash
# Single mode (formerly 'extract')
python Patch_extraction_v2.py single --image ... --label ... --grid ... --output ...

# Batch mode (NEW)
python Patch_extraction_v2.py batch --data_dir ... --grid ... --output ...

# Visualization
python Patch_extraction_v2.py visualize --output ... --split train

# Information (NEW)
python Patch_extraction_v2.py info --output ...

# Backward compatibility
python Patch_extraction_v2.py extract ...  # Alias for 'single'
```

#### Batch Naming Convention

```
data_dir/
├── Image_1.tif (or image_1.tif)
├── Image_2.tif
├── Label_1.tif (or label_1.tif)
├── Label_2.tif
├── Grid_1.shp (optional, per image)
└── Grid_2.shp
```

#### New Functions

```python
@dataclass
class ImageLabelPair:
    pair_id: str
    image_path: str
    label_path: str
    grid_path: Optional[str] = None

def find_matching_pairs(data_dir, image_pattern, label_pattern, ...) -> List[ImageLabelPair]

def extract_patches_batch(data_dir, grid_path, output_dir, ...) -> Dict
```

---

### 4. `Predict_large_image.py` → `Predict_large_image_v2.py`

#### Main Changes

| Old | New |
|-----|-----|
| No encoder support | `--encoder_name` supported |
| `metadata` format | Support both `metadata` AND `config` |
| Limited detection | Automatic encoder detection |

#### New Capabilities

```python
# Support for new checkpoint format
metadata = checkpoint.get('config', checkpoint.get('metadata', {}))

# Encoder detection
detected_encoder_name = metadata.get('encoder_name', None)

# New parameter in __init__
def __init__(self, ..., encoder_name=None, ...):

# Pass encoder to build_model
model = build_model(
    name=self.model_name,
    encoder_name=self.encoder_name,  # NEW
    in_channels=self.in_channels,
    classes=self.num_classes
)
```

---

## 🚀 Migration Guide

### For Existing Users

1. **Old scripts still work** - Backward compatibility is maintained
2. **Rename imports** if you use modules directly:
   ```python
   # Old
   from model_training import train_and_save_model
   
   # New
   from model_training_v2 import train_and_save_model_v2_full
   ```

3. **V1 checkpoints are compatible** - Predict_large_image_v2 automatically detects the format

### For New Projects

Use the `*_v2.py` files directly to benefit from all new features.

---

## 📝 Git Commands for Push

```bash
# 1. Check current status
git status

# 2. Add new v2 files
git add model_training_v2.py
git add main_v2.py  
git add Patch_extraction_v2.py
git add Predict_large_image_v2.py

# 3. Update README
git add README.md

# 4. Add CHANGELOG
git add CHANGELOG.md

# 5. Commit with descriptive message
git commit -m "feat: Major V2 update with new loss functions, encoder freezing, AMP, and batch processing

New features:
- Multiple loss functions: Focal, Dice, Tversky, Combo
- Encoder freezing for transfer learning
- Learning rate warmup
- Mixed precision training (AMP)
- Per-class IoU logging and CSV export
- Batch mode for patch extraction
- Automatic encoder detection in prediction

Updated files:
- model_training_v2.py: New TrainingConfig, loss functions, training features
- main_v2.py: Organized CLI with all new options
- Patch_extraction_v2.py: Batch mode, pattern matching, info command
- Predict_large_image_v2.py: Encoder support, format compatibility
- README.md: Complete documentation update"

# 6. Push to remote
git push origin main
```

### Alternative Commit (shorter)

```bash
git add -A
git commit -m "feat(v2): Add Focal/Tversky loss, encoder freezing, AMP, batch extraction"
git push origin main
```

---

## 📊 Improvements Summary

| Aspect | V1 | V2 |
|--------|----|----|
| **Loss functions** | Dice+BCE | 7 types (Focal, Tversky, Combo...) |
| **Transfer learning** | Basic | Encoder freezing + warmup |
| **Performance** | Standard | AMP for +40-60% speed |
| **Monitoring** | Global IoU | Per-class IoU + CSV export |
| **Patch extraction** | Single | Single + Batch + Info |
| **Prediction** | Manual config | Auto-detect encoder |
| **Configuration** | Dict | Dataclass with validation |

---

## 🔗 Files to Include in Push

```
SemanticSeg4EO/
├── model_training_v2.py     # NEW
├── main_v2.py               # NEW
├── Patch_extraction_v2.py   # NEW
├── Predict_large_image_v2.py # NEW
├── README.md                # UPDATED
├── CHANGELOG.md             # NEW
│
├── model_training.py        # Keep for backward compatibility
├── main.py                  # Keep for backward compatibility
├── Patch_extraction.py      # Keep for backward compatibility
└── Predict_large_image.py   # Keep for backward compatibility
```
