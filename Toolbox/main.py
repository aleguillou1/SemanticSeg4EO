#!/usr/bin/env python3
"""
UNIFIED SEGMENTATION TRAINING ENTRY POINT
-----------------------------------------------------------
Entry point for training both binary and multi-class segmentation models
with advanced features: Cross-Validation, Data Augmentation, etc.
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from model_training import (
    train_and_save_model,
    cross_validate_model,
    get_available_models,
    SegmentationConfig
)

def main():
    parser = argparse.ArgumentParser(
        description='UNIFIED ADVANCED SEGMENTATION TRAINING SYSTEM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Examples:
  # Binary segmentation with standard training
  python {os.path.basename(__file__)} --mode binary --dataset_root /path/to/data --model unet-dropout
  
  # Multi-class segmentation (6 classes) with cross-validation
  python {os.path.basename(__file__)} --mode multiclass --classes 6 --val_strategy kfold --n_splits 5
  
  # Binary with data augmentation and class weights
  python {os.path.basename(__file__)} --mode binary --data_augmentation --use_class_weights
  
  # Multi-class with specific encoder and pretrained weights
  python {os.path.basename(__file__)} --mode multiclass --classes 5 --encoder_name resnet50 --pretrained

Available models: {", ".join(get_available_models())}
        '''
    )
    
    # Core arguments
    parser.add_argument('--mode', type=str, default='binary', 
                       choices=['binary', 'multiclass'],
                       help='Segmentation mode: binary or multiclass (default: binary)')
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='Path to dataset root (should contain Patch folder)')
    parser.add_argument('--model', type=str, required=True, 
                       choices=get_available_models(),
                       help='Model architecture')
    
    # Task-specific arguments
    parser.add_argument('--classes', type=int, default=2,
                       help='Number of classes (for multiclass mode, default: 2)')
    parser.add_argument('--val_strategy', type=str, default='split',
                       choices=['split', 'kfold'],
                       help='Validation strategy: split or kfold (default: split)')
    
    # Data configuration
    parser.add_argument('--patch_subdir', type=str, default='Patch',
                       help='Subdirectory containing train/validation/test folders')
    parser.add_argument('--in_channels', type=int, default=10,
                       help='Number of input channels (default: 10 for Sentinel-2)')
    parser.add_argument('--patch_size', type=int, default=224,
                       help='Size of input patches (default: 224)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    
    # Model configuration
    parser.add_argument('--encoder_name', type=str, default='resnet34',
                       help='Encoder name for SMP models (default: resnet34)')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained encoder weights')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate (0.0 to 1.0, default: 0.5)')
    
    # Advanced features
    parser.add_argument('--data_augmentation', action='store_true',
                       help='Enable multi-channel data augmentation')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights to handle imbalance')
    
    # Cross-validation specific
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--subdirectories', type=str, nargs='+',
                       default=['train', 'validation', 'test'],
                       help='Subdirectories for cross-validation')
    
    # System configuration
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for training (default: cuda)')
    parser.add_argument('--save_dir', type=str, default='./trained_models',
                       help='Directory to save models (default: ./trained_models)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'multiclass' and args.classes < 2:
        print("Error: For multiclass mode, --classes must be >= 2")
        sys.exit(1)
    
    if args.mode == 'binary' and args.classes != 2:
        print("Warning: For binary mode, --classes is ignored (always 2 classes: background/foreground)")
    
    # Get configuration
    config = SegmentationConfig.get_default_config(args.mode, args.classes)
    
    # Display configuration
    print("\n" + "="*70)
    print("UNIFIED SEGMENTATION TRAINING SYSTEM")
    print("="*70)
    print(f"Configuration:")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Model: {args.model}")
    if args.mode == 'multiclass':
        print(f"  Classes: {args.classes}")
    print(f"  Validation: {args.val_strategy.upper()}")
    print(f"  Dataset: {args.dataset_root}")
    print(f"  Input Channels: {args.in_channels}")
    print(f"  Data Augmentation: {'Enabled' if args.data_augmentation else 'Disabled'}")
    print(f"  Class Weights: {'Enabled' if args.use_class_weights else 'Disabled'}")
    if args.val_strategy == 'kfold':
        print(f"  Cross-Validation Folds: {args.n_splits}")
    print("="*70 + "\n")
    
    try:
        if args.val_strategy == 'split':
            # Standard training with fixed split
            metrics = train_and_save_model(
                model_name=args.model,
                dataset_root=args.dataset_root,
                mode=args.mode,
                num_classes=args.classes,
                device=args.device,
                batch_size=args.batch_size,
                epochs=args.epochs,
                save_dir=args.save_dir,
                encoder_name=args.encoder_name,
                pretrained=args.pretrained,
                dropout_rate=args.dropout_rate,
                learning_rate=args.learning_rate,
                in_channels=args.in_channels,
                patch_size=args.patch_size,
                data_augmentation=args.data_augmentation,
                use_class_weights=args.use_class_weights,
                patch_subdir=args.patch_subdir,
                val_strategy=args.val_strategy
            )
            
            print("\n" + "="*60)
            print("STANDARD TRAINING COMPLETED")
            print("="*60)
            
            # Display results
            if args.mode == 'binary':
                print(f"Results ({args.mode}):")
                print(f"  Best Val Loss: {metrics['best_val_loss']:.4f}")
                print(f"  Best Val IoU: {metrics['best_val_iou']:.4f}")
                print(f"  Test IoU: {metrics['test_metrics']['iou']:.4f}")
                print(f"  Test F1: {metrics['test_metrics']['f1']:.4f}")
                print(f"  Test Accuracy: {metrics['test_metrics']['accuracy']:.4f}")
            else:
                print(f"Results ({args.mode}, {args.classes} classes):")
                print(f"  Best Val Loss: {metrics['best_val_loss']:.4f}")
                print(f"  Best Val mIoU: {metrics['best_val_iou']:.4f}")
                test_metrics = metrics['test_metrics']
                print(f"  Test mIoU (macro): {test_metrics.get('iou_macro', test_metrics.get('mean_iou', 0)):.4f}")
                print(f"  Test F1 (macro): {test_metrics.get('f1_macro', 0):.4f}")
                print(f"  Test Accuracy (macro): {test_metrics.get('accuracy_macro', 0):.4f}")
            
        elif args.val_strategy == 'kfold':
            # Cross-validation training
            cv_results = cross_validate_model(
                model_name=args.model,
                dataset_root=args.dataset_root,
                device=args.device,
                n_splits=args.n_splits,
                batch_size=args.batch_size,
                epochs=args.epochs,
                save_dir=args.save_dir,
                encoder_name=args.encoder_name,
                pretrained=args.pretrained,
                dropout_rate=args.dropout_rate,
                learning_rate=args.learning_rate,
                in_channels=args.in_channels,
                patch_size=args.patch_size,
                data_augmentation=args.data_augmentation,
                use_class_weights=args.use_class_weights,
                patch_subdir=args.patch_subdir,
                subdirectories=args.subdirectories,
                mode=args.mode,
                num_classes=args.classes,
                val_strategy=args.val_strategy
            )
            
            print("\n" + "="*60)
            print("CROSS-VALIDATION COMPLETED")
            print("="*60)
            
            # Display statistics
            cv_stats = cv_results['cv_stats']
            metric_name = 'IoU' if args.mode == 'binary' else 'mIoU'
            
            print(f"\nCross-Validation Statistics ({args.mode}):")
            print(f"  Mean {metric_name}: {cv_stats['mean_iou']:.4f} ± {cv_stats['std_iou']:.4f}")
            print(f"  Mean F1: {cv_stats['mean_f1']:.4f} ± {cv_stats['std_f1']:.4f}")
            print(f"  95% CI {metric_name}: [{cv_stats['ci_iou']['lower']:.4f}, {cv_stats['ci_iou']['upper']:.4f}]")
            print(f"  95% CI F1: [{cv_stats['ci_f1']['lower']:.4f}, {cv_stats['ci_f1']['upper']:.4f}]")
            
            # Interpretation
            print(f"\nInterpretation:")
            if cv_stats['std_iou'] < 0.05:
                print("  ✓ High stability across folds (consistent performance)")
            elif cv_stats['std_iou'] < 0.1:
                print("  ~ Moderate stability across folds")
            else:
                print("  ⚠ Low stability - results vary significantly across folds")
            
            if args.mode == 'binary':
                if cv_stats['mean_iou'] > 0.6:
                    print("  ✓ Good overall performance")
                elif cv_stats['mean_iou'] > 0.4:
                    print("  ~ Acceptable performance for imbalanced dataset")
                else:
                    print("  ⚠ Low performance - consider more data or different approach")
            else:
                if cv_stats['mean_iou'] > 0.5:
                    print("  ✓ Good overall performance for multi-class")
                elif cv_stats['mean_iou'] > 0.3:
                    print("  ~ Acceptable performance for multi-class")
                else:
                    print("  ⚠ Low performance - consider more data or different approach")
            
            print(f"\nResults saved to: {cv_results['cv_save_dir']}")
        
        print(f"\nAll models and metrics have been saved.")
        print("="*70)
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting tips:")
        print(f"  1. Check dataset path: {args.dataset_root}")
        print(f"  2. Verify subdirectories exist: {args.dataset_root}/{args.patch_subdir}/train/, etc.")
        print(f"  3. Check number of channels: --in_channels {args.in_channels}")
        print(f"  4. For multiclass, ensure mask values are in range [0, {args.classes-1}]")
        print(f"  5. For binary, ensure masks are binary (0/1)")
        print(f"  6. Try with --data_augmentation for small datasets")
        print(f"  7. Try with --use_class_weights for imbalanced datasets")
        
        sys.exit(1)

if __name__ == '__main__':
    main()