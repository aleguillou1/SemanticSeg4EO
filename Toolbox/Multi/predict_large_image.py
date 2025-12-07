# predict_large_image.py
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import os
import tempfile
from pathlib import Path
from tqdm import tqdm
import math
import glob
import re
from model_training import build_model
import argparse

class LargeImagePredictor:
    """Predictor for large images with patch-based processing"""
    
    def __init__(self, model_path, model_name='unet-dropout', in_channels=None, num_classes=None, 
                 patch_size=512, overlap=128, device='cuda'):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the .pth model file
            model_name: Model architecture name
            patch_size: Size of patches (recommended: 512)
            overlap: Overlap between patches (recommended: 128)
        """
        self.model_path = model_path
        self.model_name = model_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.overlap = overlap
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Configuration:")
        print(f"  - Model: {model_name}")
        print(f"  - Patch size: {patch_size}")
        print(f"  - Overlap: {overlap}")
        print(f"  - Overlap/patch ratio: {overlap/patch_size:.2%}")
        
        if overlap < patch_size // 4:
            print("Warning: Overlap may be too small, risk of artifacts!")
        
        # Load the model
        self.model = self._load_model()
        
        if self.in_channels is None:
            raise ValueError("in_channels was not properly initialized")
        if self.num_classes is None:
            raise ValueError("num_classes was not properly initialized")
            
        print(f"Final parameters:")
        print(f"  - Input channels: {self.in_channels}")
        print(f"  - Number of classes: {self.num_classes}")
        
        # Temporary directory
        self.temp_dir = tempfile.mkdtemp()
        print(f"Temporary directory: {self.temp_dir}")
    
    def _load_model(self):
        """Load the model and return detected parameters"""
        print(f"Loading model: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Detect parameters
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            model_state_dict = checkpoint['model_state_dict']
            print("Detected format: final_model (standardized)")
        elif 'model_state_dict' in checkpoint:
            metadata = checkpoint.get('metadata', {})
            model_state_dict = checkpoint['model_state_dict']
            print("Detected format: training checkpoint")
        else:
            model_state_dict = checkpoint
            metadata = {}
            print("Unknown format - attempting direct loading")
        
        # Update parameters (priority: arguments > metadata > defaults)
        detected_model_name = metadata.get('model_name', self.model_name)
        
        if detected_model_name != self.model_name:
            print(f"Detected model: {detected_model_name}, using specified: {self.model_name}")
        else:
            print(f"Model: {self.model_name}")
        
        self.in_channels = self.in_channels or metadata.get('in_channels', 4)
        self.num_classes = self.num_classes or metadata.get('num_classes', 6)
        
        print(f"  - Input channels: {self.in_channels}")
        print(f"  - Number of classes: {self.num_classes}")
        
        # Build model
        try:
            model = build_model(
                name=self.model_name,
                in_channels=self.in_channels,
                classes=self.num_classes
            )
        except Exception as e:
            print(f"Error building model {self.model_name}: {e}")
            print("Attempting with detected model from metadata...")
            model = build_model(
                name=detected_model_name,
                in_channels=self.in_channels,
                classes=self.num_classes
            )
            self.model_name = detected_model_name
            print(f"Using model: {self.model_name}")
        
        model.load_state_dict(model_state_dict)
        model.to(self.device)
        model.eval()
        
        return model
    
    def _normalize_patch(self, patch):
        """Normalize a patch"""
        patch = patch.astype(np.float32)
        if np.max(patch) > 0:
            patch = patch / np.max(patch)
        return patch
    
    def _predict_single_patch(self, patch):
        """Predict on a single patch"""
        with torch.no_grad():
            if self.in_channels is None:
                raise ValueError("self.in_channels is not defined")
                
            # Adjust channels if needed
            if patch.shape[0] != self.in_channels:
                print(f"Adjusting channels: {patch.shape[0]} -> {self.in_channels}")
                if patch.shape[0] < self.in_channels:
                    repeats = (self.in_channels + patch.shape[0] - 1) // patch.shape[0]
                    patch = np.repeat(patch, repeats, axis=0)[:self.in_channels]
                else:
                    patch = patch[:self.in_channels]
            
            input_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(self.device)
            output = self.model(input_tensor)
            
            if self.num_classes > 1:
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
            else:
                prediction = (torch.sigmoid(output) > 0.5).long()
                
            return prediction.squeeze().cpu().numpy().astype(np.uint8)
    
    def _extract_patches_with_grid(self, image_path):
        """Extract patches with regular grid and border handling"""
        with rasterio.open(image_path) as src:
            profile = src.profile
            height, width = src.height, src.width
            
            print(f"Image dimensions: {width}x{height}")
            print(f"Source channels: {src.count}")
            
            # Calculate grid
            step = self.patch_size - self.overlap
            n_patches_x = max(1, math.ceil((width - self.overlap) / step))
            n_patches_y = max(1, math.ceil((height - self.overlap) / step))
            
            # Adjust to cover entire image
            total_width = (n_patches_x - 1) * step + self.patch_size
            total_height = (n_patches_y - 1) * step + self.patch_size
            
            if total_width < width:
                n_patches_x += 1
            if total_height < height:
                n_patches_y += 1
            
            print(f"Grid: {n_patches_x}x{n_patches_y} = {n_patches_x * n_patches_y} patches")
            print(f"Step: {step} pixels")
            
            patches_info = []
            
            for i in tqdm(range(n_patches_y), desc="Extracting patches"):
                for j in range(n_patches_x):
                    x = j * step
                    y = i * step
                    
                    # Final border adjustment
                    if x + self.patch_size > width:
                        x = width - self.patch_size
                    if y + self.patch_size > height:
                        y = height - self.patch_size
                    x, y = max(0, x), max(0, y)
                    
                    window = Window(x, y, self.patch_size, self.patch_size)
                    patch = src.read(window=window)
                    
                    patches_info.append({
                        'patch': patch,
                        'window': window,
                        'position': (i, j),
                        'coords': (x, y)
                    })
            
            return patches_info, profile, (height, width)
    
    def _reconstruct_simple_majority(self, predicted_patches, original_shape):
        """Simple majority vote reconstruction"""
        height, width = original_shape
        
        vote_count = np.zeros((height, width, self.num_classes), dtype=np.int32)
        
        print("Reconstructing with majority voting...")
        
        for patch_info in tqdm(predicted_patches, desc="Counting votes"):
            pred_mask = patch_info['predicted_mask']
            window = patch_info['window']
            x, y = int(window.col_off), int(window.row_off)
            
            for class_idx in range(self.num_classes):
                vote_count[y:y+self.patch_size, x:x+self.patch_size, class_idx] += (pred_mask == class_idx)
        
        reconstruction = np.argmax(vote_count, axis=2).astype(np.uint8)
        
        max_votes = np.max(vote_count, axis=2)
        total_votes = np.sum(vote_count, axis=2)
        confidence = max_votes / np.maximum(total_votes, 1)
        
        print(f"Average confidence: {np.mean(confidence):.3f}")
        print(f"Low confidence areas (<0.5): {np.sum(confidence < 0.5) / confidence.size:.2%}")
        
        return reconstruction
    
    def _reconstruct_from_patches(self, predicted_patches, original_profile, original_shape, output_path):
        """Final reconstruction"""
        height, width = original_shape
        
        print("Reconstructing with majority vote...")
        reconstruction = self._reconstruct_simple_majority(predicted_patches, original_shape)
        
        # Check coverage
        unique, counts = np.unique(reconstruction, return_counts=True)
        total_pixels = np.sum(counts)
        
        print("Class coverage:")
        for cls, count in zip(unique, counts):
            percentage = (count / total_pixels) * 100
            print(f"  - Class {cls}: {percentage:.2f}%")
        
        # Save
        output_profile = original_profile.copy()
        output_profile.update({
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(reconstruction, 1)
        
        print(f"Image saved: {output_path}")
        
        unique, counts = np.unique(reconstruction, return_counts=True)
        print("Final result:")
        for cls, count in zip(unique, counts):
            percentage = (count / total_pixels) * 100
            print(f"  - Class {cls}: {count} pixels ({percentage:.2f}%)")
    
    def predict_large_image(self, input_image_path, output_path):
        """Main prediction function"""
        print("Starting large image prediction")
        print("=" * 50)
        
        patches_info, profile, original_shape = self._extract_patches_with_grid(input_image_path)
        
        print("Predicting patches...")
        for patch_info in tqdm(patches_info, desc="Predicting patches"):
            patch = patch_info['patch']
            normalized_patch = self._normalize_patch(patch)
            predicted_mask = self._predict_single_patch(normalized_patch)
            patch_info['predicted_mask'] = predicted_mask
            
            unique = np.unique(predicted_mask)
            if len(unique) > 0 and len(patches_info) <= 100:
                print(f"  Patch {patch_info['position']}: classes {unique}")
        
        self._reconstruct_from_patches(patches_info, profile, original_shape, output_path)
        
        import shutil
        shutil.rmtree(self.temp_dir)
        
        print("Prediction completed!")

def get_available_models():
    """Return list of available models"""
    return [
        'unet-dropout', 'unet', 'deeplabv3', 'deeplabv3+', 'fpn', 
        'pspnet', 'manet', 'linknet', 'pan'
    ]

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='Multi-class prediction on large images with patch processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Usage examples:
  python {os.path.basename(__file__)} --model_path /path/to/model.pth --input image.tif --output prediction.tif
  python {os.path.basename(__file__)} --model_path /path/to/model.pth --model_name deeplabv3+ --input image.tif --output prediction.tif
  python {os.path.basename(__file__)} --model_path /path/to/model.pth --input image.tif --output prediction.tif --patch_size 256 --overlap 64

Available models: {", ".join(get_available_models())}
        '''
    )
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to .pth model file')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--output', type=str, required=True, 
                       help='Path for output image')
    
    # Optional model arguments
    parser.add_argument('--model_name', type=str, default='unet-dropout', 
                       choices=get_available_models(),
                       help=f"Model architecture name (default: unet-dropout)")
    parser.add_argument('--in_channels', type=int, default=None, 
                       help='Number of input channels (auto-detected if not specified)')
    parser.add_argument('--num_classes', type=int, default=None, 
                       help='Number of classes (auto-detected if not specified)')
    
    # Optional processing arguments
    parser.add_argument('--patch_size', type=int, default=512, 
                       help='Patch size (default: 512)')
    parser.add_argument('--overlap', type=int, default=128, 
                       help='Overlap between patches (default: 128)')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'],
                       help='Device for inference (default: cuda)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model_path):
        print(f"Error: Model {args.model_path} does not exist")
        return
    
    if not os.path.exists(args.input):
        print(f"Error: Input image {args.input} does not exist")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print("Starting prediction")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Architecture: {args.model_name}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Parameters: patch_size={args.patch_size}, overlap={args.overlap}")
    print("=" * 60)
    
    try:
        predictor = LargeImagePredictor(
            model_path=args.model_path,
            model_name=args.model_name,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            overlap=args.overlap,
            device=args.device
        )
        
        predictor.predict_large_image(args.input, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTips:")
        print("1. Check if the model is compatible with the specified architecture")
        print("2. Try with different --model_name")
        print("3. Manually specify --in_channels and --num_classes")
        print("4. Reduce --patch_size if memory issues")

# Simplified function for import
def predict_large_image(model_path, input_image, output_image, model_name='unet-dropout', **kwargs):
    """
    Simple function to predict large images without artifacts
    
    Args:
        model_path: Path to model file
        input_image: Input image path
        output_image: Output image path
        model_name: Architecture name ('unet-dropout', 'deeplabv3+', etc.)
        **kwargs: Additional parameters
    """
    predictor = LargeImagePredictor(
        model_path=model_path,
        model_name=model_name,
        **kwargs
    )
    predictor.predict_large_image(input_image, output_image)

if __name__ == "__main__":
    main()