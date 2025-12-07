# predict_large_image_binary.py
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import os
import tempfile
from pathlib import Path
from tqdm import tqdm
import math
from model_training_binary import build_model

class LargeImagePredictorBinary:
    """Predictor for large images with patch-based processing for binary segmentation"""
    
    def __init__(self, model_path, model_name=None, in_channels=10, patch_size=224, 
                 overlap=64, threshold=0.5, device='cuda'):
        """
        Initialize the predictor for large images
        
        Args:
            model_path: Path to .pth model file
            model_name: Model name (auto-detected if None)
            in_channels: Number of input channels
            patch_size: Size of square patches
            overlap: Overlap between patches (in pixels)
            threshold: Threshold for binary segmentation
            device: Device for inference ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.model_name = model_name
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.overlap = overlap
        self.threshold = threshold
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load the model
        self.model = self._load_model()
        
        # Temporary directory for patches
        self.temp_dir = tempfile.mkdtemp()
        print(f"Temporary directory: {self.temp_dir}")
    
    def _load_model(self):
        """Load model with automatic parameter detection"""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Automatic parameter detection
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            model_state_dict = checkpoint['model_state_dict']
            self.model_name = self.model_name or metadata.get('model_name', 'unet-dropout')
            self.in_channels = metadata.get('in_channels', self.in_channels)
            print("Detected format: final_model (standardized)")
            
        elif 'model_state_dict' in checkpoint and 'val_loss' in checkpoint:
            metadata = checkpoint
            model_state_dict = checkpoint['model_state_dict']
            self.model_name = self.model_name or 'unet-dropout'
            print("Detected format: training checkpoint")
        else:
            model_state_dict = checkpoint
            self.model_name = self.model_name or 'unet-dropout'
            print("Unknown format - using default values")
        
        print(f"Loading model: {self.model_name}")
        print(f"  - Channels: {self.in_channels}")
        print(f"  - Device: {self.device}")
        print(f"  - Patch size: {self.patch_size}")
        print(f"  - Overlap: {self.overlap}")
        print(f"  - Threshold: {self.threshold}")
        
        # Build model
        model = build_model(
            name=self.model_name,
            in_channels=self.in_channels,
            classes=1  # Always 1 for binary
        )
        
        model.load_state_dict(model_state_dict)
        model.to(self.device)
        model.eval()
        
        return model
    
    def _normalize_patch(self, patch):
        """Normalize a patch as during training"""
        patch = patch.astype(np.float32)
        if np.max(patch) > 0:
            # Same normalization as during training
            p99 = np.percentile(patch, 99)
            patch = np.clip(patch / p99, 0, 1)
        else:
            patch = np.zeros_like(patch)
        return patch
    
    def _predict_single_patch(self, patch):
        """Predict on a single patch"""
        with torch.no_grad():
            # Prepare input
            if patch.shape[0] != self.in_channels:
                if patch.shape[0] < self.in_channels:
                    repeats = self.in_channels // patch.shape[0] + 1
                    patch = np.tile(patch, (repeats, 1, 1))[:self.in_channels]
                else:
                    patch = patch[:self.in_channels]
            
            input_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(self.device)
            output = self.model(input_tensor)
            
            # Binary conversion
            if output.shape[1] == 1:
                probabilities = torch.sigmoid(output)
                predicted_mask = (probabilities > self.threshold).squeeze().cpu().numpy().astype(np.uint8)
            else:
                prediction = torch.argmax(output, dim=1)
                predicted_mask = prediction.squeeze().cpu().numpy().astype(np.uint8)
            
            return predicted_mask
    
    def _extract_patches(self, image_path):
        """Extract patches from large image"""
        with rasterio.open(image_path) as src:
            profile = src.profile
            height, width = src.height, src.width
            
            print(f"Image dimensions: {width}x{height}")
            
            # Calculate number of patches
            step = self.patch_size - self.overlap
            n_patches_x = math.ceil((width - self.overlap) / step)
            n_patches_y = math.ceil((height - self.overlap) / step)
            
            print(f"Splitting into {n_patches_x}x{n_patches_y} = {n_patches_x * n_patches_y} patches")
            
            patches_info = []
            
            for i in tqdm(range(n_patches_y), desc="Extracting patches"):
                for j in range(n_patches_x):
                    # Calculate patch position
                    x = j * step
                    y = i * step
                    
                    # Border adjustment
                    if x + self.patch_size > width:
                        x = width - self.patch_size
                    if y + self.patch_size > height:
                        y = height - self.patch_size
                    
                    # Read patch
                    window = Window(x, y, self.patch_size, self.patch_size)
                    patch = src.read(window=window)
                    
                    # Save patch information
                    patch_info = {
                        'patch': patch,
                        'window': window,
                        'position': (i, j),
                        'coords': (x, y)
                    }
                    patches_info.append(patch_info)
            
            return patches_info, profile, (height, width)
    
    def _reconstruct_from_patches(self, predicted_patches, original_profile, original_shape, output_path):
        """Reconstruct complete image from predicted patches"""
        height, width = original_shape
        
        # Create empty image for reconstruction
        reconstructed = np.zeros((height, width), dtype=np.uint8)
        weight_matrix = np.zeros((height, width), dtype=np.float32)
        
        print("Reconstructing image...")
        
        for patch_info in tqdm(predicted_patches, desc="Reconstruction"):
            pred_mask = patch_info['predicted_mask']
            window = patch_info['window']
            x, y = int(window.col_off), int(window.row_off)
            h, w = pred_mask.shape
            
            # Create weight mask (gaussian or linear for smoothing edges)
            weight_patch = self._create_weight_mask(h, w)
            
            # Add patch with weighting
            reconstructed[y:y+h, x:x+w] += (pred_mask * weight_patch).astype(np.uint8)
            weight_matrix[y:y+h, x:x+w] += weight_patch
        
        # Normalize by dividing by weight matrix
        reconstructed = np.round(reconstructed / (weight_matrix + 1e-8)).astype(np.uint8)
        
        # Save reconstructed image
        output_profile = original_profile.copy()
        output_profile.update({
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(reconstructed, 1)
        
        print(f"Reconstructed image saved: {output_path}")
        
        # Statistics
        unique_vals, counts = np.unique(reconstructed, return_counts=True)
        total_pixels = np.sum(counts)
        print("Reconstructed image statistics:")
        for val, count in zip(unique_vals, counts):
            percentage = (count / total_pixels) * 100
            class_name = "Class 1 (positive)" if val == 1 else "Class 0 (negative)"
            print(f"  - {class_name}: {count} pixels ({percentage:.2f}%)")
    
    def _create_weight_mask(self, height, width):
        """Create weight mask to smooth transitions between patches"""
        # Simple linear mask
        weight = np.ones((height, width), dtype=np.float32)
        
        # Reduce weight on edges
        border = self.overlap // 2
        if border > 0:
            for i in range(border):
                factor = (i + 1) / (border + 1)
                weight[i, :] *= factor  # Top border
                weight[height-1-i, :] *= factor  # Bottom border
                weight[:, i] *= factor  # Left border
                weight[:, width-1-i] *= factor  # Right border
        
        return weight
    
    def predict_large_image(self, input_image_path, output_path, save_patches=False):
        """
        Predict on large image by splitting into patches
        
        Args:
            input_image_path: Path to input image
            output_path: Path for output image
            save_patches: If True, save individual patches
        """
        print("Starting large image prediction")
        print("=" * 60)
        
        # Step 1: Extract patches
        patches_info, profile, original_shape = self._extract_patches(input_image_path)
        
        # Step 2: Predict on each patch
        print("Predicting patches...")
        predicted_patches = []
        
        for patch_info in tqdm(patches_info, desc="Predicting patches"):
            patch = patch_info['patch']
            
            # Normalize patch
            normalized_patch = self._normalize_patch(patch)
            
            # Predict
            predicted_mask = self._predict_single_patch(normalized_patch)
            
            # Save individual patch if requested
            if save_patches:
                patch_path = os.path.join(self.temp_dir, 
                                        f"patch_{patch_info['position'][0]}_{patch_info['position'][1]}.tif")
                with rasterio.open(patch_path, 'w', **profile) as dst:
                    dst.write(predicted_mask, 1)
            
            patch_info['predicted_mask'] = predicted_mask
            predicted_patches.append(patch_info)
        
        # Step 3: Reconstruct image
        self._reconstruct_from_patches(predicted_patches, profile, original_shape, output_path)
        
        # Cleanup
        if not save_patches:
            import shutil
            shutil.rmtree(self.temp_dir)
            print("Temporary directory cleaned")
        else:
            print(f"Patches saved in: {self.temp_dir}")
        
        print("Prediction completed successfully!")

def main():
    """Main function with simple interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prediction on large image with patch splitting')
    parser.add_argument('--model', type=str, required=True, help='Path to .pth model')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path for output image')
    parser.add_argument('--model_name', type=str, default=None, help='Model name (auto-detected if not specified)')
    parser.add_argument('--in_channels', type=int, default=10, help='Number of input channels')
    parser.add_argument('--patch_size', type=int, default=224, help='Patch size')
    parser.add_argument('--overlap', type=int, default=10, help='Overlap between patches')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
    parser.add_argument('--save_patches', action='store_true', help='Save individual patches')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = LargeImagePredictorBinary(
        model_path=args.model,
        model_name=args.model_name,
        in_channels=args.in_channels,
        patch_size=args.patch_size,
        overlap=args.overlap,
        threshold=args.threshold
    )
    
    # Run prediction
    predictor.predict_large_image(
        input_image_path=args.input,
        output_path=args.output,
        save_patches=args.save_patches
    )

# Simple usage (without command line)
def predict_simple(model_path, input_image, output_image, **kwargs):
    """
    Simple function to predict large image
    
    Args:
        model_path: Path to model
        input_image: Input image path
        output_image: Output image path
        **kwargs: Optional parameters (patch_size, overlap, threshold, etc.)
    """
    predictor = LargeImagePredictorBinary(model_path, **kwargs)
    predictor.predict_large_image(input_image, output_image)

if __name__ == "__main__":
    # Command line usage example:
    # python predict_large_image_binary.py --model /path/model.pth --input /path/image.tif --output /path/prediction.tif
    
    # Simple usage example:
    # predict_simple("model.pth", "large_image.tif", "prediction.tif", patch_size=256, overlap=32)
    
    main()