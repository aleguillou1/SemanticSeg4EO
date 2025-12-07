# inference_binary.py
import torch
import numpy as np
import rasterio
import os
import glob
import re
from pathlib import Path
from comparative_code_binary_da_s2 import build_model


def clean_model_name(model_name):
    """Clean model name for use in filenames"""
    # Remove special characters and spaces
    cleaned = re.sub(r'[^\w\s-]', '', model_name)
    cleaned = re.sub(r'[-\s]+', '_', cleaned)
    return cleaned.lower()


def find_model_files(model_dir, model_name):
    """Automatically find model files in directory"""
    model_files = {}
    
    patterns = [
        f"{model_name}_best_iou.pth",
        f"{model_name}_best_loss.pth", 
        f"{model_name}_final_model.pth",
        f"{model_name}_best_miou.pth",  # Alternative naming
        f"best_iou.pth",  # Fallback patterns
        f"best_loss.pth",
        f"final_model.pth"
    ]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(model_dir, pattern))
        if matches:
            key = pattern.replace(f"{model_name}_", "").replace(".pth", "")
            model_files[key] = matches[0]
            print(f"✅ Model found: {key} -> {matches[0]}")
    
    return model_files


def predict_and_save_georef_binary(model_path, image_path, output_path=None, model_name=None, in_channels=None, threshold=0.7):
    """Binary segmentation inference function"""
    
    # Auto-generate output name if not provided
    if output_path is None:
        image_name = Path(image_path).stem
        model_type = Path(model_path).stem
        output_dir = Path(model_path).parent
        clean_name = clean_model_name(model_name) if model_name else "model"
        output_path = os.path.join(output_dir, f"pred_{image_name}_{clean_name}_{model_type}.tif")
        print(f"📁 Auto-generated output: {output_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # AUTOMATIC MODEL TYPE DETECTION
    if 'metadata' in checkpoint:
        metadata = checkpoint['metadata']
        model_state_dict = checkpoint['model_state_dict']
        print("✅ Format detected: final_model (standardized)")
        
    elif 'model_state_dict' in checkpoint and 'val_loss' in checkpoint:
        metadata = checkpoint
        model_state_dict = checkpoint['model_state_dict']
        
        if 'best_loss' in checkpoint:
            print(f"✅ Format detected: best_loss (loss={checkpoint['best_loss']:.4f})")
        elif 'best_iou' in checkpoint:
            print(f"✅ Format detected: best_iou (IoU={checkpoint['best_iou']:.4f})")
        else:
            print("✅ Format detected: training checkpoint")
            
    else:
        model_state_dict = checkpoint
        metadata = {}
        print("⚠️  Unknown format - attempting direct loading")
    
    # MODEL PARAMETERS
    final_model_name = model_name or metadata.get('model_name', 'unet-alg')
    final_in_channels = in_channels or metadata.get('in_channels', 10)
    
    print(f"🔧 Model configuration (BINARY):")
    print(f"   - Name: {final_model_name}")
    print(f"   - Input channels: {final_in_channels}")
    print(f"   - Type: Binary segmentation")
    print(f"   - Threshold: {threshold}")
    
    # Build model for BINARY segmentation (classes=1)
    model = build_model(
        name=final_model_name,
        in_channels=final_in_channels,
        classes=1
    )
    
    # Load weights
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Load image and its georeferencing
    with rasterio.open(image_path) as src:
        img = src.read()
        profile = src.profile
        
        print(f"📷 Image loaded: {img.shape}")
        print(f"📐 Profile: {profile}")
        
        # Normalize image
        img = img.astype(np.float32)
        if np.max(img) > 0:
            p99 = np.percentile(img, 99)
            img = np.clip(img / p99, 0, 1)
        else:
            img = np.zeros_like(img)
        
        # Check channels
        if img.shape[0] != final_in_channels:
            print(f"⚠️  Adjusting channels: {img.shape[0]} -> {final_in_channels}")
            if img.shape[0] < final_in_channels:
                repeats = final_in_channels // img.shape[0] + 1
                img = np.tile(img, (repeats, 1, 1))[:final_in_channels]
            else:
                img = img[:final_in_channels]
        
        print(f"🔧 Image after preprocessing: {img.shape}")
        print(f"📊 Value range: [{img.min():.3f}, {img.max():.3f}]")
        
        # BINARY prediction
        with torch.no_grad():
            input_tensor = torch.from_numpy(img).unsqueeze(0).float()
            output = model(input_tensor)
            
            if output.shape[1] == 1:
                probabilities = torch.sigmoid(output)
                predicted_mask = (probabilities > threshold).squeeze().numpy().astype(np.uint8)
            else:
                print("⚠️  Multi-class output detected, using argmax")
                prediction = torch.argmax(output, dim=1)
                predicted_mask = prediction.squeeze().numpy().astype(np.uint8)
        
        # Save with same georeferencing
        profile.update({
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(predicted_mask, 1)
    
    # Prediction statistics
    unique_vals, counts = np.unique(predicted_mask, return_counts=True)
    total_pixels = np.sum(counts)
    print(f"📊 Prediction statistics:")
    for val, count in zip(unique_vals, counts):
        percentage = (count / total_pixels) * 100
        class_name = "Class 1 (positive)" if val == 1 else "Class 0 (negative)"
        print(f"   - {class_name}: {count} pixels ({percentage:.2f}%)")
    
    print(f"💾 Georeferenced binary prediction saved: {output_path}")
    
    return predicted_mask


def batch_predict_binary(model_dir, model_name, images_dir, output_dir, in_channels=None, threshold=0.5):
    """Batch prediction for multiple images with automatic model detection"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Automatically find all models
    model_files = find_model_files(model_dir, model_name)
    
    if not model_files:
        print(f"❌ No models found for {model_name} in {model_dir}")
        return []
    
    # Find all .tif images
    image_paths = glob.glob(os.path.join(images_dir, "*.tif"))
    image_paths.extend(glob.glob(os.path.join(images_dir, "*.tiff")))
    
    print(f"🔍 {len(image_paths)} images found in {images_dir}")
    print(f"🔍 {len(model_files)} models found: {list(model_files.keys())}")
    
    all_results = []
    
    # Process each model
    for model_type, model_path in model_files.items():
        print(f"\n{'='*60}")
        print(f"🚀 PROCESSING WITH MODEL: {model_type}")
        print(f"{'='*60}")
        
        model_results = []
        
        for image_path in image_paths:
            try:
                image_name = Path(image_path).stem
                # Auto-generate output name with model name
                clean_model = clean_model_name(model_name)
                output_path = os.path.join(output_dir, f"pred_{image_name}_{clean_model}_{model_type}.tif")
                
                print(f"\n🎯 Processing: {image_name} with {model_name} ({model_type})")
                mask = predict_and_save_georef_binary(
                    model_path=model_path,
                    image_path=image_path,
                    output_path=output_path,
                    model_name=model_name,
                    in_channels=in_channels,
                    threshold=threshold
                )
                
                model_results.append({
                    'image': image_name,
                    'model_name': model_name,
                    'model_type': model_type,
                    'output_path': output_path,
                    'mask_stats': np.unique(mask, return_counts=True)
                })
                
            except Exception as e:
                print(f"❌ Error with {image_path} and {model_type}: {e}")
                continue
        
        print(f"✅ {model_type}: {len(model_results)}/{len(image_paths)} images processed")
        all_results.extend(model_results)
    
    print(f"\n🎉 BATCH PROCESSING COMPLETE: {len(all_results)} predictions generated")
    return all_results


def predict_single_image(model_dir, model_name, image_path, output_dir=None, model_types=None, threshold=0.5, in_channels=None):
    """Prediction for single image with model selection"""
    
    if output_dir is None:
        output_dir = model_dir
    
    # Automatically find models
    model_files = find_model_files(model_dir, model_name)
    
    if not model_files:
        print(f"❌ No models found for {model_name} in {model_dir}")
        return []
    
    # Filter models if specified
    if model_types is not None:
        model_files = {k: v for k, v in model_files.items() if k in model_types}
        print(f"🔧 Selected models: {list(model_files.keys())}")
    
    image_name = Path(image_path).stem
    results = []
    
    print(f"\n🎯 STARTING PREDICTION FOR: {image_name}")
    print(f"📁 Model: {model_name}")
    print(f"📷 Image: {image_path}")
    print(f"📁 Output: {output_dir}")
    
    for model_type, model_path in model_files.items():
        try:
            # Auto-generate output name with model name
            clean_model = clean_model_name(model_name)
            output_path = os.path.join(output_dir, f"pred_{image_name}_{clean_model}_{model_type}.tif")
            
            print(f"\n{'='*50}")
            print(f"🚀 PREDICTION WITH {model_name.upper()} - {model_type.upper()}")
            print(f"{'='*50}")
            
            mask = predict_and_save_georef_binary(
                model_path=model_path,
                image_path=image_path,
                output_path=output_path,
                model_name=model_name,
                in_channels=in_channels,
                threshold=threshold
            )
            
            results.append({
                'model_name': model_name,
                'model_type': model_type,
                'output_path': output_path,
                'mask': mask
            })
            
        except Exception as e:
            print(f"❌ Error with model {model_type}: {e}")
            continue
    
    print(f"\n🎉 PREDICTION COMPLETE: {len(results)}/{len(model_files)} models processed")
    return results


# SIMPLIFIED USAGE EXAMPLES
if __name__ == "__main__":
    
    # Base paths - ADAPT THESE
    model_dir = "/mnt/c/Users/PC/Bureau/Travail/Thèse_SSD/Unet_RS2/Schorre/nouveau_label/en_grand/testasuppr"
    image_path = "/mnt/c/Users/PC/Bureau/Travail/Thèse_SSD/Unet_RS2/Schorre/nouveau_label/en_grand/Patch/test/images/patch_398.0.tif"
    
    model_name = "unet++"  # Just the model name
    
    # 1. SIMPLE PREDICTION FOR SINGLE IMAGE
    print("🎯 SIMPLE PREDICTION FOR SINGLE IMAGE")
    results = predict_single_image(
        model_dir=model_dir,
        model_name=model_name,
        image_path=image_path,
        threshold=0.3,
        in_channels=10  # Optional - auto-detected if omitted
    )
    
    # 2. BATCH PREDICTION (optional)
    print("\n" + "="*60)
    print("🔄 BATCH PREDICTION")
    print("="*60)
    
    # Uncomment to use batch prediction
    # batch_results = batch_predict_binary(
    #     model_dir=model_dir,
    #     model_name=model_name,
    #     images_dir="/path/to/your/images",
    #     output_dir=f"{model_dir}/predictions_batch",
    #     threshold=0.3,
    #     in_channels=10
    # )
    
    print("\n🎉 ALL BINARY PREDICTIONS COMPLETE!")
    
    # Display generated files
    if results:
        print("\n📁 GENERATED FILES:")
        for result in results:
            print(f"   ✅ {Path(result['output_path']).name}")
    
    # Threshold adjustment tips
    print("\n💡 THRESHOLD ADJUSTMENT TIPS:")
    print("   - threshold=0.3 → More sensitive (more true positives, risk of false positives)")
    print("   - threshold=0.5 → Default balance")
    print("   - threshold=0.7 → More conservative (fewer false positives, risk of false negatives)")