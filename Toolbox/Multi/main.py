# train_advanced.py
from model_training import train_and_save_model, get_available_models

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train segmentation model on .tif dataset')
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

    args = parser.parse_args()

    print("\n" + "="*50)
    print("Model Training")
    print("="*50 + "\n")

    try:
        metrics = train_and_save_model(
            model_name=args.model,
            dataset_root=args.dataset_root,
            device=args.device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            save_dir=args.save_dir,
            encoder_name=args.encoder_name,
            pretrained=args.pretrained,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate
        )

        print("\n" + "="*50)
        print("Training Summary")
        print("="*50)
        
        # Safely access metrics
        model_name = metrics.get('model_name', 'N/A')
        encoder_name = metrics.get('encoder_name', 'N/A')
        
        # Complexity info
        complexity_info = metrics.get('complexity_info', {})
        parameters = complexity_info.get('parameters', 'N/A')
        inference_time = complexity_info.get('inference_time', 'N/A')
        
        # Performance metrics
        test_metrics = metrics.get('test_metrics', {})
        test_miou = test_metrics.get('iou_macro', 'N/A')
        test_f1 = test_metrics.get('f1_macro', 'N/A')
        test_accuracy = test_metrics.get('accuracy_macro', 'N/A')
        
        # Best scores
        best_val_loss = metrics.get('best_val_loss', 'N/A')
        best_val_miou = metrics.get('best_val_miou', 'N/A')
        
        print(f"Model: {model_name}")
        print(f"Encoder: {encoder_name}")
        if parameters != 'N/A':
            print(f"Parameters: {parameters:,}")
        else:
            print(f"Parameters: {parameters}")
            
        print(f"Inference Time: {inference_time}s")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Best Val mIoU: {best_val_miou:.4f}")
        print(f"Test mIoU (macro): {test_miou:.4f}")
        print(f"Test F1 (macro): {test_f1:.4f}")
        print(f"Test Accuracy (macro): {test_accuracy:.4f}")
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()