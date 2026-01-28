#!/usr/bin/env python3
"""
UNIFIED SEGMENTATION TRAINING ENTRY POINT V2
-----------------------------------------------------------
Enhanced CLI with all new features:
- Multiple loss functions (Focal, Dice, Tversky, Combo)
- Encoder freezing for transfer learning
- Learning rate warmup
- Mixed precision training (AMP)
- Per-class metrics logging
- CSV export
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from model_training_v2 import (
    train_and_save_model_v2,
    train_and_save_model_v2_full,
    cross_validate_model_v2,
    get_available_models,
    get_available_losses,
    TrainingConfig
)


def main():
    parser = argparse.ArgumentParser(
        description='UNIFIED SEGMENTATION TRAINING SYSTEM V2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
================================================================================
EXAMPLES
================================================================================

# Basic multi-class training with Focal Loss
python {os.path.basename(__file__)} \\
    --mode multiclass --classes 5 \\
    --dataset_root /path/to/data \\
    --model unet++ \\
    --loss_type focal

# Training with encoder freezing (first 5 epochs)
python {os.path.basename(__file__)} \\
    --mode multiclass --classes 5 \\
    --dataset_root /path/to/data \\
    --model unet++ \\
    --freeze_encoder --freeze_epochs 5

# Full featured training
python {os.path.basename(__file__)} \\
    --mode multiclass --classes 5 \\
    --dataset_root /path/to/data \\
    --model unet++ \\
    --loss_type focal_dice \\
    --freeze_encoder --freeze_epochs 3 \\
    --warmup_epochs 2 \\
    --use_amp \\
    --log_per_class \\
    --class_names background water vegetation buildings roads

# Binary segmentation with Dice loss
python {os.path.basename(__file__)} \\
    --mode binary \\
    --dataset_root /path/to/data \\
    --model unet \\
    --loss_type dice

================================================================================
AVAILABLE OPTIONS
================================================================================
Models: {", ".join(get_available_models())}
Losses: {", ".join(get_available_losses())}

Loss descriptions:
  ce          - Cross Entropy only
  dice        - Dice Loss only
  dice_ce     - Dice + Cross Entropy (default, balanced)
  focal       - Focal Loss (good for imbalanced classes)
  focal_dice  - Focal + Dice (recommended for severe imbalance)
  tversky     - Tversky Loss (control FP/FN tradeoff)
  combo       - CE + Dice + Focal combined
================================================================================
        '''
    )
    
    # ==================== CORE ARGUMENTS ====================
    core = parser.add_argument_group('Core Settings')
    core.add_argument('--mode', type=str, default='multiclass',
                      choices=['binary', 'multiclass'],
                      help='Segmentation mode (default: multiclass)')
    core.add_argument('--dataset_root', type=str, required=True,
                      help='Path to dataset root (should contain Patch folder)')
    core.add_argument('--model', type=str, required=True,
                      choices=get_available_models(),
                      help='Model architecture')
    core.add_argument('--classes', type=int, default=5,
                      help='Number of classes for multiclass mode (default: 5)')
    core.add_argument('--val_strategy', type=str, default='split',
                      choices=['split', 'kfold'],
                      help='Validation strategy: split or kfold (default: split)')
    
    # ==================== DATA ARGUMENTS ====================
    data = parser.add_argument_group('Data Configuration')
    data.add_argument('--patch_subdir', type=str, default='Patch',
                      help='Subdirectory containing train/validation/test')
    data.add_argument('--in_channels', type=int, default=3,
                      help='Number of input channels (default: 3)')
    data.add_argument('--patch_size', type=int, default=224,
                      help='Size of input patches (default: 224)')
    data.add_argument('--data_augmentation', action='store_true',
                      help='Enable data augmentation')
    data.add_argument('--no_data_augmentation', action='store_true',
                      help='Disable data augmentation')
    
    # ==================== TRAINING ARGUMENTS ====================
    training = parser.add_argument_group('Training Configuration')
    training.add_argument('--epochs', type=int, default=100,
                          help='Number of training epochs (default: 100)')
    training.add_argument('--batch_size', type=int, default=8,
                          help='Batch size (default: 8)')
    training.add_argument('--learning_rate', type=float, default=5e-4,
                          help='Learning rate (default: 5e-4)')
    training.add_argument('--patience', type=int, default=15,
                          help='Early stopping patience (default: 15)')
    
    # ==================== MODEL ARGUMENTS ====================
    model_args = parser.add_argument_group('Model Configuration')
    model_args.add_argument('--encoder_name', type=str, default='resnet34',
                            help='Encoder backbone (default: resnet34)')
    model_args.add_argument('--pretrained', action='store_true', default=True,
                            help='Use pretrained encoder (default: True)')
    model_args.add_argument('--no_pretrained', action='store_true',
                            help='Do not use pretrained encoder')
    model_args.add_argument('--dropout_rate', type=float, default=0.3,
                            help='Dropout rate (default: 0.3)')
    
    # ==================== LOSS ARGUMENTS (NEW) ====================
    loss_args = parser.add_argument_group('Loss Configuration (NEW)')
    loss_args.add_argument('--loss_type', type=str, default='dice_ce',
                           choices=get_available_losses(),
                           help='Loss function type (default: dice_ce)')
    loss_args.add_argument('--use_class_weights', action='store_true', default=True,
                           help='Use class weights for imbalance (default: True)')
    loss_args.add_argument('--no_class_weights', action='store_true',
                           help='Disable class weights')
    loss_args.add_argument('--focal_gamma', type=float, default=2.0,
                           help='Focal loss gamma parameter (default: 2.0)')
    loss_args.add_argument('--focal_alpha', type=float, default=0.25,
                           help='Focal loss alpha parameter (default: 0.25)')
    loss_args.add_argument('--tversky_alpha', type=float, default=0.7,
                           help='Tversky loss alpha (FP weight, default: 0.7)')
    loss_args.add_argument('--tversky_beta', type=float, default=0.3,
                           help='Tversky loss beta (FN weight, default: 0.3)')
    loss_args.add_argument('--dice_weight', type=float, default=0.5,
                           help='Dice component weight in combo losses (default: 0.5)')
    
    # ==================== ENCODER FREEZING (NEW) ====================
    freeze_args = parser.add_argument_group('Encoder Freezing (NEW)')
    freeze_args.add_argument('--freeze_encoder', action='store_true',
                             help='Freeze encoder for initial epochs')
    freeze_args.add_argument('--freeze_epochs', type=int, default=5,
                             help='Number of epochs to keep encoder frozen (default: 5)')
    
    # ==================== LEARNING RATE SCHEDULE (NEW) ====================
    lr_args = parser.add_argument_group('Learning Rate Schedule (NEW)')
    lr_args.add_argument('--scheduler_type', type=str, default='reduce_plateau',
                         choices=['reduce_plateau', 'cosine', 'one_cycle'],
                         help='LR scheduler type (default: reduce_plateau)')
    lr_args.add_argument('--warmup_epochs', type=int, default=0,
                         help='Number of warmup epochs (default: 0)')
    lr_args.add_argument('--warmup_lr', type=float, default=1e-6,
                         help='Initial LR during warmup (default: 1e-6)')
    
    # ==================== MIXED PRECISION (NEW) ====================
    amp_args = parser.add_argument_group('Mixed Precision Training (NEW)')
    amp_args.add_argument('--use_amp', action='store_true',
                          help='Enable Automatic Mixed Precision (faster training)')
    
    # ==================== LOGGING (NEW) ====================
    log_args = parser.add_argument_group('Logging & Metrics (NEW)')
    log_args.add_argument('--log_per_class', action='store_true', default=True,
                          help='Log per-class IoU metrics (default: True)')
    log_args.add_argument('--no_log_per_class', action='store_true',
                          help='Disable per-class logging')
    log_args.add_argument('--save_csv', action='store_true', default=True,
                          help='Save training logs to CSV (default: True)')
    log_args.add_argument('--class_names', type=str, nargs='+', default=None,
                          help='Names for each class (e.g., --class_names bg water veg)')
    
    # ==================== CROSS-VALIDATION ====================
    cv_args = parser.add_argument_group('Cross-Validation')
    cv_args.add_argument('--n_splits', type=int, default=5,
                         help='Number of folds for K-Fold CV (default: 5)')
    cv_args.add_argument('--subdirectories', type=str, nargs='+',
                         default=['train', 'validation', 'test'],
                         help='Subdirectories to include in CV')
    
    # ==================== SYSTEM ARGUMENTS ====================
    system = parser.add_argument_group('System Configuration')
    system.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for training (default: cuda)')
    system.add_argument('--save_dir', type=str, default='./trained_models',
                        help='Directory to save models (default: ./trained_models)')
    
    args = parser.parse_args()
    
    # ==================== RESOLVE CONFLICTING FLAGS ====================
    # Data augmentation
    if args.no_data_augmentation:
        data_augmentation = False
    elif args.data_augmentation:
        data_augmentation = True
    else:
        data_augmentation = True  # Default to True
    
    # Pretrained
    pretrained = not args.no_pretrained
    
    # Class weights
    use_class_weights = not args.no_class_weights
    
    # Per-class logging
    log_per_class = not args.no_log_per_class
    
    # ==================== VALIDATE ARGUMENTS ====================
    if args.mode == 'multiclass' and args.classes < 2:
        print("Error: For multiclass mode, --classes must be >= 2")
        sys.exit(1)
    
    if args.class_names and len(args.class_names) != args.classes:
        print(f"Warning: Number of class names ({len(args.class_names)}) doesn't match number of classes ({args.classes})")
        print("Class names will be auto-generated.")
        args.class_names = None
    
    # ==================== DISPLAY CONFIGURATION ====================
    print("\n" + "="*70)
    print("UNIFIED SEGMENTATION TRAINING SYSTEM V2")
    print("="*70)
    print(f"\n[Core]")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Classes: {args.classes}")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset_root}")
    
    print(f"\n[Training]")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Data augmentation: {'Yes' if data_augmentation else 'No'}")
    
    print(f"\n[Loss]")
    print(f"  Type: {args.loss_type}")
    print(f"  Class weights: {'Yes' if use_class_weights else 'No'}")
    if 'focal' in args.loss_type:
        print(f"  Focal gamma: {args.focal_gamma}")
    if 'tversky' in args.loss_type:
        print(f"  Tversky alpha/beta: {args.tversky_alpha}/{args.tversky_beta}")
    
    print(f"\n[Advanced]")
    print(f"  Encoder freezing: {'Yes (' + str(args.freeze_epochs) + ' epochs)' if args.freeze_encoder else 'No'}")
    print(f"  LR warmup: {'Yes (' + str(args.warmup_epochs) + ' epochs)' if args.warmup_epochs > 0 else 'No'}")
    print(f"  Mixed precision (AMP): {'Yes' if args.use_amp else 'No'}")
    print(f"  Per-class metrics: {'Yes' if log_per_class else 'No'}")
    print(f"  Validation strategy: {args.val_strategy.upper()}")
    if args.val_strategy == 'kfold':
        print(f"  K-Fold splits: {args.n_splits}")
    
    if args.class_names:
        print(f"  Class names: {', '.join(args.class_names)}")
    
    print("="*70 + "\n")
    
    # ==================== RUN TRAINING ====================
    try:
        if args.val_strategy == 'split':
            # Standard training with fixed split
            metrics = train_and_save_model_v2_full(
                model_name=args.model,
                dataset_root=args.dataset_root,
                mode=args.mode,
                num_classes=args.classes,
                device=args.device,
                batch_size=args.batch_size,
                epochs=args.epochs,
                save_dir=args.save_dir,
                encoder_name=args.encoder_name,
                pretrained=pretrained,
                dropout_rate=args.dropout_rate,
                learning_rate=args.learning_rate,
                in_channels=args.in_channels,
                patch_size=args.patch_size,
                data_augmentation=data_augmentation,
                use_class_weights=use_class_weights,
                patch_subdir=args.patch_subdir,
                # V2 options
                loss_type=args.loss_type,
                freeze_encoder=args.freeze_encoder,
                freeze_epochs=args.freeze_epochs,
                warmup_epochs=args.warmup_epochs,
                use_amp=args.use_amp,
                log_per_class_metrics=log_per_class,
                class_names=args.class_names,
                focal_gamma=args.focal_gamma,
                tversky_alpha=args.tversky_alpha
            )
            
            print("\n" + "="*70)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("="*70)
            
            print(f"\nResults saved to: {args.save_dir}")
            print(f"  - Model: {metrics['model_path']}")
            print(f"  - Training plot: {args.save_dir}/{args.model}_{args.mode}_training_plot.png")
            print(f"  - CSV log: {args.save_dir}/{args.model}_training_log.csv")
            print(f"  - JSON metrics: {metrics.get('json_path', 'N/A')}")
        
        elif args.val_strategy == 'kfold':
            # K-Fold Cross-Validation
            cv_results = cross_validate_model_v2(
                model_name=args.model,
                dataset_root=args.dataset_root,
                device=args.device,
                n_splits=args.n_splits,
                batch_size=args.batch_size,
                epochs=args.epochs,
                save_dir=args.save_dir,
                encoder_name=args.encoder_name,
                pretrained=pretrained,
                dropout_rate=args.dropout_rate,
                learning_rate=args.learning_rate,
                in_channels=args.in_channels,
                patch_size=args.patch_size,
                data_augmentation=data_augmentation,
                use_class_weights=use_class_weights,
                patch_subdir=args.patch_subdir,
                subdirectories=args.subdirectories,
                mode=args.mode,
                num_classes=args.classes,
                # V2 options
                loss_type=args.loss_type,
                freeze_encoder=args.freeze_encoder,
                freeze_epochs=args.freeze_epochs,
                log_per_class_metrics=log_per_class,
                class_names=args.class_names
            )
            
            print("\n" + "="*70)
            print("CROSS-VALIDATION COMPLETED SUCCESSFULLY")
            print("="*70)
            
            cv_stats = cv_results['cv_stats']
            print(f"\nFinal Results ({args.n_splits}-Fold CV):")
            print(f"  mIoU: {cv_stats['mean_iou']:.4f} ± {cv_stats['std_iou']:.4f}")
            print(f"  F1:   {cv_stats['mean_f1']:.4f} ± {cv_stats['std_f1']:.4f}")
            print(f"  95% CI mIoU: [{cv_stats['ci_iou']['lower']:.4f}, {cv_stats['ci_iou']['upper']:.4f}]")
            
            if 'per_class' in cv_stats:
                print(f"\nPer-class results:")
                for class_name, stats in cv_stats['per_class'].items():
                    print(f"  {class_name}: {stats['mean_iou']:.4f} ± {stats['std_iou']:.4f}")
            
            print(f"\nResults saved to: {cv_results['cv_save_dir']}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*70)
        print("TROUBLESHOOTING TIPS")
        print("="*70)
        print(f"1. Check dataset path: {args.dataset_root}")
        print(f"2. Verify folders exist: {args.dataset_root}/{args.patch_subdir}/train/, validation/, test/")
        print(f"3. Check input channels: --in_channels {args.in_channels}")
        print(f"4. For multiclass, mask values should be in [0, {args.classes-1}]")
        print(f"5. Try --loss_type focal_dice for severely imbalanced data")
        print(f"6. Try --freeze_encoder --freeze_epochs 5 for better transfer learning")
        print("="*70)
        
        sys.exit(1)


if __name__ == '__main__':
    main()
