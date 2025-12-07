# predict_single_patch.py
import torch
import numpy as np
import rasterio
import os
import glob
from model_training import build_model
import re

def find_model_files(models_dir, model_name):
    """Automatically find model files in directory"""
    model_files = {}
    
    patterns = {
        'best_miou': f"*{model_name}*best*miou*.pth",
        'best_loss': f"*{model_name}*best*loss*.pth", 
        'final_model': f"*{model_name}*final*.pth"
    }
    
    for key, pattern in patterns.items():
        matches = glob.glob(os.path.join(models_dir, pattern))
        if matches:
            model_files[key] = matches[0]
            print(f"Found {key}: {os.path.basename(matches[0])}")
        else:
            print(f"Not found {key} with pattern: {pattern}")
    
    return model_files

def extract_patch_number(image_path):
    """Extract patch number from filename"""
    filename = os.path.basename(image_path)
    
    patterns = [
        r'patch[_-]?(\d+\.?\d*)',  # patch_884.0, patch884, patch-884
        r'(\d+\.?\d*)(?=\.tif)',    # 884.0 in patch_884.0.tif
        r'(\d+)'                    # simple digits
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            patch_num = match.group(1)
            clean_patch_num = patch_num.replace('.', '_')
            print(f"Extracted patch number: {patch_num} -> {clean_patch_num}")
            return clean_patch_num
    
    print("Patch number not found, using 'unknown'")
    return "unknown"

def predict_and_save_georef(model_path, image_path, output_path, model_name=None, in_channels=None, num_classes=None):
    """Universal function for all .pth model types"""
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Detect model type
    if 'metadata' in checkpoint:
        metadata = checkpoint['metadata']
        model_state_dict = checkpoint['model_state_dict']
        print("Detected format: final_model (standardized)")
        
    elif 'model_state_dict' in checkpoint and 'val_loss' in checkpoint:
        metadata = checkpoint.get('metadata', {})
        model_state_dict = checkpoint['model_state_dict']
        
        if 'best_loss' in checkpoint:
            print(f"Detected format: best_loss (loss={checkpoint['best_loss']:.4f})")
        elif 'best_miou' in checkpoint:
            print(f"Detected format: best_miou (mIoU={checkpoint['best_miou']:.4f})")
        else:
            print("Detected format: training checkpoint")
            
    else:
        model_state_dict = checkpoint
        metadata = {}
        print("Unknown format - attempting direct loading")
    
    # Model parameters (priority: arguments > metadata > defaults)
    final_model_name = model_name or metadata.get('model_name', 'unet')
    final_in_channels = in_channels or metadata.get('in_channels', 4)
    final_num_classes = num_classes or metadata.get('num_classes', 6)
    
    print(f"Model configuration:")
    print(f"  - Name: {final_model_name}")
    print(f"  - Input channels: {final_in_channels}")
    print(f"  - Classes: {final_num_classes}")
    
    # Build model
    model = build_model(
        name=final_model_name,
        in_channels=final_in_channels,
        classes=final_num_classes
    )
    
    # Load weights
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Load image with georeferencing
    with rasterio.open(image_path) as src:
        img = src.read()
        profile = src.profile
        
        print(f"Loaded image: {img.shape}")
        
        # Normalize image
        img = img.astype(np.float32)
        if np.max(img) > 0:
            img = img / np.max(img)
        else:
            img = np.zeros_like(img)
        
        # Check channels
        if img.shape[0] != final_in_channels:
            print(f"Adjusting channels: {img.shape[0]} -> {final_in_channels}")
            if img.shape[0] < final_in_channels:
                img = np.repeat(img, final_in_channels, axis=0)[:final_in_channels]
            else:
                img = img[:final_in_channels]
        
        # Multi-class prediction
        with torch.no_grad():
            input_tensor = torch.from_numpy(img).unsqueeze(0).float()
            output = model(input_tensor)
            
            if final_num_classes > 1:
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
            else:
                prediction = (torch.sigmoid(output) > 0.5).long()
                
            predicted_mask = prediction.squeeze().numpy().astype(np.uint8)
        
        # Save with same georeferencing
        profile.update({
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(predicted_mask, 1)
    
    print(f"Georeferenced prediction saved: {output_path}")
    print(f"Predicted mask - Classes: {np.unique(predicted_mask)}")
    
    return predicted_mask

def run_complete_inference(model_name, models_dir, image_path, output_dir=None):
    """
    Main automated function
    
    Args:
        model_name: Model name (e.g., 'deeplabv3+')
        models_dir: Directory containing models
        image_path: Path to patch image
        output_dir: Output directory (optional)
    """
    
    if output_dir is None:
        output_dir = models_dir
    
    patch_number = extract_patch_number(image_path)
    
    model_files = find_model_files(models_dir, model_name)
    
    if not model_files:
        print("No models found! Check model name and directory.")
        return
    
    results = {}
    
    for model_type, model_path in model_files.items():
        print(f"\n" + "="*60)
        print(f"PREDICTION WITH {model_type.upper()}")
        print("="*60)
        
        output_filename = f"{model_name}_{model_type}_patch_{patch_number}.tif"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            predicted_mask = predict_and_save_georef(
                model_path=model_path,
                image_path=image_path,
                output_path=output_path,
                model_name=model_name
            )
            results[model_type] = {
                'output_path': output_path,
                'mask': predicted_mask
            }
            
        except Exception as e:
            print(f"Error with {model_type}: {e}")
            continue
    
    return results

if __name__ == "__main__":
    
    # Minimal configuration
    MODEL_NAME = "unet"
    MODELS_DIR = "/path/to/models"
    IMAGE_PATH = "/path/to/patch_image.tif"
    
    print("Starting automated inference")
    print("="*50)
    print(f"Model: {MODEL_NAME}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Image: {IMAGE_PATH}")
    print("="*50)
    
    results = run_complete_inference(
        model_name=MODEL_NAME,
        models_dir=MODELS_DIR, 
        image_path=IMAGE_PATH
    )
    
    if results:
        print("\nAll predictions completed!")
        print("Generated files:")
        for model_type, result in results.items():
            print(f"  - {model_type}: {result['output_path']}")
    else:
        print("\nNo predictions were made.")