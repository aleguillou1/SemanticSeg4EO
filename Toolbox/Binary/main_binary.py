# main_binary.py
from model_training_binary import train_and_save_binary_model, get_available_models

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train advanced BINARY segmentation model on .tif dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--model', type=str, required=True, choices=get_available_models(), help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda', help='Device: "cuda" or "cpu"')
    parser.add_argument('--save_dir', type=str, default='./trained_models', help='Directory to save model')
    parser.add_argument('--encoder_name', type=str, default='resnet34', help='Encoder name for SMP models')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained encoder')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate (0.0 to 1.0)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    
    # PARAMETERS FOR BINARY
    parser.add_argument('--in_channels', type=int, default=10, help='Number of input channels in images (default: 10 for Sentinel-2)')
    parser.add_argument('--patch_size', type=int, default=224, help='Size of input patches (height and width)')
    
    # NEW: Option for data augmentation
    parser.add_argument('--data_augmentation', action='store_true', help='Enable multi-channel data augmentation for training')
    
    # NEW: Option for class weights
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights to handle imbalance')

    args = parser.parse_args()

    print("\n" + "="*50)
    print("Advanced BINARY Segmentation Model Training")
    print("="*50)
    print(f"Configuration:")
    print(f"  - Model: {args.model}")
    print(f"  - Input Channels: {args.in_channels}")
    print(f"  - Data Augmentation: {'Enabled' if args.data_augmentation else 'Disabled'}")
    print(f"  - Class Weights: {'Enabled' if args.use_class_weights else 'Disabled'}")
    print(f"  - Learning Rate: {args.learning_rate}")
    print(f"  - Epochs: {args.epochs}")
    print("="*50 + "\n")

    try:
        metrics = train_and_save_binary_model(
            model_name=args.model,
            dataset_root=args.dataset_root,
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
        )

        print("\n" + "="*50)
        print("BINARY TRAINING SUMMARY")
        print("="*50)
        
        # Safe access to metrics
        model_name = metrics.get('model_name', 'N/A')
        encoder_name = metrics.get('encoder_name', 'N/A')
        in_channels = metrics.get('in_channels', 'N/A')
        task_type = metrics.get('task_type', 'N/A')
        patch_size = metrics.get('patch_size', 'N/A')
        data_augmentation = metrics.get('data_augmentation', 'N/A')
        use_class_weights = metrics.get('use_class_weights', 'N/A')
        
        # Complexity
        complexity_info = metrics.get('complexity_info', {})
        parameters = complexity_info.get('parameters', 'N/A')
        inference_time = complexity_info.get('inference_time', 'N/A')
        
        # Performance metrics
        test_metrics = metrics.get('test_metrics', {})
        test_iou = test_metrics.get('iou', 'N/A')
        test_f1 = test_metrics.get('f1', 'N/A')
        test_accuracy = test_metrics.get('accuracy', 'N/A')
        test_precision = test_metrics.get('precision', 'N/A')
        test_recall = test_metrics.get('recall', 'N/A')
        
        # Best scores
        best_val_loss = metrics.get('best_val_loss', 'N/A')
        best_val_iou = metrics.get('best_val_iou', 'N/A')
        
        print(f"Model: {model_name}")
        print(f"Encoder: {encoder_name}")
        print(f"Input Channels: {in_channels}")
        print(f"Task Type: {task_type}")
        print(f"Patch Size: {patch_size}")
        print(f"Data Augmentation: {data_augmentation}")
        print(f"Class Weights: {use_class_weights}")
        
        if parameters != 'N/A':
            print(f"Parameters: {parameters:,}")
        else:
            print(f"Parameters: {parameters}")
            
        print(f"Inference Time: {inference_time:.4f}s")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Best Val IoU: {best_val_iou:.4f}")
        print(f"Test IoU: {test_iou:.4f}")
        print(f"Test F1: {test_f1:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        
        # Performance interpretation
        print("\n" + "="*50)
        print("PERFORMANCE ANALYSIS")
        print("="*50)
        if test_iou != 'N/A' and test_recall != 'N/A':
            if test_recall > 0.9 and test_precision < 0.3:
                print("Warning: Model predicts too many false positives (High recall, low precision)")
                print("  → Try increasing threshold or adjusting class weights")
            elif test_iou < 0.3:
                print("Insufficient performance: IoU too low")
                print("  → Check class imbalance and data quality")
            elif test_iou > 0.6:
                print("Good performance: Satisfactory IoU")
            else:
                print("Average performance: Possible improvement")
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("  - Check dataset path (--dataset_root)")
        print("  - Ensure images and masks have the same names")
        print("  - Check number of channels (--in_channels)")
        print("  - For small dataset, use --data_augmentation and --use_class_weights")