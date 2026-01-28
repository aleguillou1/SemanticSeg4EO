# predict_large_image_hybrid.py - VERSION CORRIGÉE ET AMÉLIORÉE

"""
Universal predictor for large images with seamless segmentation.
Handles both binary and multi-class segmentation with robust preprocessing.
"""

import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import os
import tempfile
from pathlib import Path
from tqdm import tqdm
import math
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

# Try to import model building function
try:
    from model_training import build_model_for_prediction as build_model
    print("✓ Using model_training for model building")
except ImportError as e:
    print(f"✗ Error importing from model_training: {e}")
    print("Please ensure model_training.py is in the same directory or in PYTHONPATH")
    sys.exit(1)


class LargeImagePredictorHybrid:
    """
    Universal predictor for large images with seamless reconstruction.
    Handles both binary and multi-class segmentation with robust preprocessing.
    """
    
    def __init__(self, model_path, model_name=None, encoder_name=None, in_channels=None, 
                 num_classes=None, patch_size=512, overlap=128, 
                 device='cuda', threshold=0.5, normalization_percentile=99,
                 background_value=0, normalize_per_channel=True):
        """
        Initialize the hybrid predictor
        
        Args:
            model_path: Path to .pth model file
            model_name: Model architecture name
            encoder_name: Encoder backbone name (e.g., resnet34, resnet50, efficientnet-b0)
            in_channels: Number of input channels
            num_classes: Number of output classes
            patch_size: Size of square patches
            overlap: Overlap between patches (in pixels)
            device: Device for inference
            threshold: Threshold for binary segmentation
            normalization_percentile: Percentile for normalization
            background_value: Value for background class
            normalize_per_channel: Whether to normalize each channel separately
        """
        self.model_path = model_path
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.overlap = overlap
        self.threshold = threshold
        self.normalization_percentile = normalization_percentile
        self.background_value = background_value
        self.normalize_per_channel = normalize_per_channel
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        print(f"\n{'='*70}")
        print("UNIVERSAL LARGE IMAGE PREDICTOR INITIALIZATION")
        print(f"{'='*70}")
        print(f"  Model path: {model_path}")
        print(f"  Device: {self.device}")
        print(f"  Patch size: {patch_size}")
        print(f"  Overlap: {overlap}")
        print(f"  Normalization: {normalization_percentile}th percentile")
        print(f"  Per-channel normalization: {normalize_per_channel}")
        print(f"  Background value: {background_value}")
        
        # Validate parameters
        if overlap < patch_size // 4:
            print(f"\n⚠ Warning: Overlap ({overlap}) is less than patch_size/4 ({patch_size//4})")
            print("  Consider increasing overlap for better reconstruction")
        
        # Load the model
        self.model = self._load_model()
        
        # Determine mode based on num_classes
        self.mode = 'binary' if self.num_classes == 1 else 'multiclass'
        
        print(f"\nFinal configuration:")
        print(f"  - Model: {self.model_name}")
        print(f"  - Encoder: {self.encoder_name}")
        print(f"  - Input channels: {self.in_channels}")
        print(f"  - Number of classes: {self.num_classes}")
        print(f"  - Mode: {self.mode.upper()}")
        if self.mode == 'binary':
            print(f"  - Threshold: {self.threshold}")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="predictor_")
        print(f"  - Temporary directory: {self.temp_dir}")
    
    def _load_model(self):
        """Load model with automatic parameter detection"""
        print(f"\nLoading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # Try to extract model parameters
        model_state_dict = None
        metadata = {}
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
                # Support both 'metadata' (old format) and 'config' (model_training_v2 format)
                metadata = checkpoint.get('config', checkpoint.get('metadata', {}))
                print("✓ Detected format: training checkpoint")
            elif 'state_dict' in checkpoint:
                model_state_dict = checkpoint['state_dict']
                print("✓ Detected format: state_dict only")
            else:
                # Try direct loading
                model_state_dict = checkpoint
                print("⚠ Unknown dict format - attempting direct loading")
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")
        
        # Extract metadata
        detected_model_name = metadata.get('model_name', None)
        detected_encoder_name = metadata.get('encoder_name', None)
        detected_in_channels = metadata.get('in_channels', None)
        detected_num_classes = metadata.get('num_classes', None)
        
        # Try to infer from state_dict keys if not in metadata
        if detected_in_channels is None:
            # Look for conv1 weight shape
            for key in model_state_dict.keys():
                if 'conv1.weight' in key or 'conv1.conv.weight' in key:
                    weight_shape = model_state_dict[key].shape
                    if len(weight_shape) == 4:
                        detected_in_channels = weight_shape[1]
                        break
        
        if detected_num_classes is None:
            # Look for final layer weight shape
            for key in model_state_dict.keys():
                if any(x in key for x in ['conv_cls.weight', 'classifier.weight', 'last_layer.weight']):
                    weight_shape = model_state_dict[key].shape
                    if len(weight_shape) == 4:
                        detected_num_classes = weight_shape[0]
                        break
        
        print(f"\nDetected parameters:")
        print(f"  - Model name: {detected_model_name or 'Not detected'}")
        print(f"  - Encoder name: {detected_encoder_name or 'Not detected'}")
        print(f"  - Input channels: {detected_in_channels or 'Not detected'}")
        print(f"  - Number of classes: {detected_num_classes or 'Not detected'}")
        
        # Set parameters (user args take precedence)
        if self.model_name is None:
            self.model_name = detected_model_name or 'unet'
        
        if self.encoder_name is None:
            self.encoder_name = detected_encoder_name or 'resnet34'
            if detected_encoder_name is None:
                print(f"⚠ Encoder not detected, using default: {self.encoder_name}")
        
        if self.in_channels is None:
            self.in_channels = detected_in_channels or 10
            if detected_in_channels is None:
                print(f"⚠ Input channels not detected, using default: {self.in_channels}")
        
        if self.num_classes is None:
            self.num_classes = detected_num_classes or 1
            if detected_num_classes is None:
                print(f"⚠ Number of classes not detected, assuming binary: {self.num_classes}")
        
        # Build model
        print(f"\nBuilding model: {self.model_name}")
        print(f"  - Encoder: {self.encoder_name}")
        print(f"  - Input channels: {self.in_channels}")
        print(f"  - Output classes: {self.num_classes}")
        
        try:
            model = build_model(
                name=self.model_name,
                encoder_name=self.encoder_name,
                in_channels=self.in_channels,
                classes=self.num_classes
            )
        except Exception as e:
            print(f"✗ Error building model: {e}")
            print("\nTrying alternative approaches...")
            
            # Try with common model names
            for alt_name in ['unet', 'unet-dropout', 'deeplabv3', 'deeplabv3+', 'fpn']:
                try:
                    print(f"  Trying with model name: {alt_name}")
                    model = build_model(
                        name=alt_name,
                        encoder_name=self.encoder_name,
                        in_channels=self.in_channels,
                        classes=self.num_classes
                    )
                    self.model_name = alt_name
                    print(f"✓ Success with {alt_name}")
                    break
                except:
                    continue
            else:
                raise ValueError(f"Could not build model with any architecture")
        
        # Load weights
        try:
            # Remove 'module.' prefix if present (from DataParallel)
            if any(key.startswith('module.') for key in model_state_dict.keys()):
                model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
            
            model.load_state_dict(model_state_dict)
            print("✓ Model weights loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Error loading state dict: {e}")
            print("Attempting to load with strict=False...")
            try:
                model.load_state_dict(model_state_dict, strict=False)
                print("✓ Model weights loaded with strict=False")
            except Exception as e2:
                print(f"✗ Critical error: {e2}")
                raise
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _normalize_patch(self, patch):
        """
        Normalize patch similar to training pipeline
        
        Args:
            patch: Input patch (C, H, W)
        
        Returns:
            Normalized patch (C, H, W)
        """
        patch = patch.astype(np.float32)
        
        if self.normalize_per_channel:
            # Normalize each channel separately
            normalized = np.zeros_like(patch)
            for c in range(patch.shape[0]):
                channel = patch[c]
                if np.max(channel) > 0:
                    # Robust normalization with percentile
                    p_val = np.percentile(channel, self.normalization_percentile)
                    if p_val > 0:
                        channel_norm = np.clip(channel / p_val, 0, 1)
                    else:
                        channel_norm = np.zeros_like(channel)
                else:
                    channel_norm = np.zeros_like(channel)
                normalized[c] = channel_norm
        else:
            # Global normalization
            if np.max(patch) > 0:
                p_val = np.percentile(patch, self.normalization_percentile)
                if p_val > 0:
                    normalized = np.clip(patch / p_val, 0, 1)
                else:
                    normalized = np.zeros_like(patch)
            else:
                normalized = np.zeros_like(patch)
        
        return normalized
    
    def _create_weight_mask(self, height, width):
        """
        Create weight mask for seamless blending
        
        Args:
            height: Mask height
            width: Mask width
        
        Returns:
            Weight mask with linear falloff at edges
        """
        weight = np.ones((height, width), dtype=np.float32)
        
        if self.overlap > 0:
            border = min(self.overlap // 2, min(height, width) // 2)
            
            if border > 0:
                # Linear falloff from edge to center
                for i in range(border):
                    # Weight increases from edge to center
                    factor = (i + 1) / border
                    
                    # Apply to borders
                    weight[i, :] *= factor  # Top
                    weight[height-1-i, :] *= factor  # Bottom
                    weight[:, i] *= factor  # Left
                    weight[:, width-1-i] *= factor  # Right
        
        return weight
    
    def _predict_single_patch(self, patch):
        """
        Predict probabilities for a single patch
        
        Args:
            patch: Normalized patch (C, H, W)
        
        Returns:
            Probabilities tensor
        """
        with torch.no_grad():
            # Ensure correct number of channels
            if patch.shape[0] != self.in_channels:
                if patch.shape[0] < self.in_channels:
                    # Repeat last channel if needed
                    repeats = math.ceil(self.in_channels / patch.shape[0])
                    patch = np.repeat(patch, repeats, axis=0)[:self.in_channels]
                else:
                    # Take first n channels
                    patch = patch[:self.in_channels]
            
            # Add batch dimension and convert to tensor
            input_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(self.device)
            
            # Forward pass
            output = self.model(input_tensor)
            
            # Handle different output formats
            if isinstance(output, dict):
                output = output['out'] if 'out' in output else output['logits']
            elif isinstance(output, tuple):
                output = output[0]
            
            # Apply appropriate activation
            if self.num_classes == 1:
                # Binary: sigmoid
                probabilities = torch.sigmoid(output)
                return probabilities.squeeze().cpu().numpy()  # (H, W)
            else:
                # Multi-class: softmax
                probabilities = torch.softmax(output, dim=1)
                return probabilities.squeeze(0).cpu().numpy()  # (C, H, W)
    
    def _extract_patches(self, image_path):
        """
        Extract patches from large image with grid
        
        Args:
            image_path: Path to input image
        
        Returns:
            patches_info, profile, original_shape, nodata_value
        """
        with rasterio.open(image_path) as src:
            profile = src.profile.copy()
            height, width = src.height, src.width
            num_channels = src.count
            nodata_value = profile.get('nodata', None)
            
            print(f"\nImage information:")
            print(f"  Size: {width} x {height} pixels")
            print(f"  Channels: {num_channels}")
            print(f"  Data type: {src.dtypes[0]}")
            print(f"  Nodata value: {nodata_value}")
            
            if num_channels != self.in_channels:
                print(f"⚠ Warning: Image has {num_channels} channels, model expects {self.in_channels}")
                if num_channels < self.in_channels:
                    print(f"  Will repeat channels to match model input")
            
            # Calculate grid
            step = self.patch_size - self.overlap
            step = max(step, 1)
            
            n_patches_x = max(1, math.ceil(width / step))
            n_patches_y = max(1, math.ceil(height / step))
            
            # Adjust to ensure coverage
            actual_width = (n_patches_x - 1) * step + self.patch_size
            actual_height = (n_patches_y - 1) * step + self.patch_size
            
            if actual_width < width:
                n_patches_x += 1
            if actual_height < height:
                n_patches_y += 1
            
            print(f"\nPatch extraction:")
            print(f"  Grid: {n_patches_x} x {n_patches_y} = {n_patches_x * n_patches_y} patches")
            print(f"  Step: {step} pixels")
            
            patches_info = []
            
            # Extract patches
            for i in tqdm(range(n_patches_y), desc="Extracting patches"):
                for j in range(n_patches_x):
                    # Calculate position
                    x = j * step
                    y = i * step
                    
                    # Adjust for borders
                    if x + self.patch_size > width:
                        x = width - self.patch_size
                    if y + self.patch_size > height:
                        y = height - self.patch_size
                    
                    x, y = max(0, x), max(0, y)
                    
                    # Read patch
                    window = Window(x, y, self.patch_size, self.patch_size)
                    patch = src.read(window=window)
                    
                    # Check for nodata
                    has_nodata = False
                    if nodata_value is not None:
                        has_nodata = np.any(patch == nodata_value)
                    
                    patches_info.append({
                        'patch': patch,
                        'window': window,
                        'coords': (x, y),
                        'has_nodata': has_nodata
                    })
            
            return patches_info, profile, (height, width), nodata_value
    
    def _reconstruct_image(self, predicted_patches, original_shape):
        """
        Reconstruct image using weighted blending
        
        Args:
            predicted_patches: List of patches with predictions
            original_shape: (height, width) of original image
        
        Returns:
            reconstructed_mask, confidence_map
        """
        height, width = original_shape
        
        if self.mode == 'binary':
            return self._reconstruct_binary(predicted_patches, height, width)
        else:
            return self._reconstruct_multiclass(predicted_patches, height, width)
    
    def _reconstruct_binary(self, predicted_patches, height, width):
        """
        Reconstruct binary segmentation
        """
        # Initialize accumulation arrays
        weighted_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        
        print("Reconstructing binary mask...")
        
        for patch_info in tqdm(predicted_patches, desc="Blending patches"):
            probs = patch_info['probs']  # (H, W) probabilities
            window = patch_info['window']
            x, y = int(window.col_off), int(window.row_off)
            h, w = probs.shape
            
            # Create weight mask
            weight_mask = self._create_weight_mask(h, w)
            
            # Accumulate
            weighted_sum[y:y+h, x:x+w] += probs * weight_mask
            weight_sum[y:y+h, x:x+w] += weight_mask
        
        # Avoid division by zero
        weight_sum = np.maximum(weight_sum, 1e-8)
        
        # Final probability map
        final_probs = weighted_sum / weight_sum
        
        # Convert to binary mask
        foreground_value = 1 if self.background_value == 0 else 0
        binary_mask = np.where(final_probs > self.threshold, 
                              foreground_value, 
                              self.background_value).astype(np.uint8)
        
        # Confidence map
        confidence = np.where(final_probs > self.threshold, 
                             final_probs, 
                             1 - final_probs)
        
        return binary_mask, confidence
    
    def _reconstruct_multiclass(self, predicted_patches, height, width):
        """
        Reconstruct multi-class segmentation
        """
        # Initialize accumulation arrays
        weighted_sums = np.zeros((self.num_classes, height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        
        print(f"Reconstructing {self.num_classes}-class mask...")
        
        for patch_info in tqdm(predicted_patches, desc="Blending patches"):
            probs = patch_info['probs']  # (C, H, W) probabilities
            window = patch_info['window']
            x, y = int(window.col_off), int(window.row_off)
            h, w = probs.shape[1], probs.shape[2]
            
            # Create weight mask
            weight_mask = self._create_weight_mask(h, w)
            
            # Accumulate for each class
            for c in range(self.num_classes):
                class_probs = probs[c]
                weighted_sums[c, y:y+h, x:x+w] += class_probs * weight_mask
            
            weight_sum[y:y+h, x:x+w] += weight_mask
        
        # Normalize
        weight_sum = np.maximum(weight_sum, 1e-8)
        for c in range(self.num_classes):
            weighted_sums[c] /= weight_sum
        
        # Final class assignment (argmax)
        class_mask = np.argmax(weighted_sums, axis=0).astype(np.uint8)
        
        # Confidence map (max probability)
        confidence = np.max(weighted_sums, axis=0)
        
        return class_mask, confidence
    
    def predict_large_image(self, input_image_path, output_path, 
                           save_confidence=False, save_patches=False,
                           output_nodata=255):
        """
        Main prediction function for large images
        
        Args:
            input_image_path: Path to input image
            output_path: Path for output segmentation
            save_confidence: Save confidence map
            save_patches: Save individual patches
            output_nodata: Nodata value for output
        """
        print(f"\n{'='*70}")
        print("STARTING PREDICTION")
        print(f"{'='*70}")
        
        # Step 1: Extract patches
        print(f"\n[1/4] Extracting patches...")
        patches_info, profile, original_shape, input_nodata = self._extract_patches(input_image_path)
        
        # Step 2: Predict on patches
        print(f"\n[2/4] Predicting patches...")
        for patch_info in tqdm(patches_info, desc="Predicting"):
            patch = patch_info['patch']
            
            # Handle nodata if present
            if patch_info['has_nodata'] and input_nodata is not None:
                # Create mask for valid pixels
                valid_mask = np.all(patch != input_nodata, axis=0)
                patch_info['valid_mask'] = valid_mask
            else:
                patch_info['valid_mask'] = None
            
            # Normalize patch
            normalized = self._normalize_patch(patch)
            
            # Get probabilities
            probs = self._predict_single_patch(normalized)
            patch_info['probs'] = probs
            
            # Save patch if requested
            if save_patches:
                self._save_patch(patch_info, profile, output_nodata)
        
        # Step 3: Reconstruct with weighted blending
        print(f"\n[3/4] Reconstructing image...")
        final_mask, confidence = self._reconstruct_image(patches_info, original_shape)
        
        # Handle nodata from source
        if input_nodata is not None:
            print("Applying source nodata mask...")
            with rasterio.open(input_image_path) as src:
                first_band = src.read(1)
                nodata_mask = (first_band == input_nodata)
                if np.any(nodata_mask):
                    final_mask[nodata_mask] = output_nodata
                    if save_confidence:
                        confidence[nodata_mask] = 0
        
        # Step 4: Save results
        print(f"\n[4/4] Saving results...")
        
        # Save main mask
        output_profile = profile.copy()
        output_profile.update({
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw',
            'nodata': output_nodata
        })
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(final_mask, 1)
        
        print(f"✓ Segmentation saved: {output_path}")
        
        # Save confidence map if requested
        if save_confidence:
            base_name = os.path.splitext(output_path)[0]
            confidence_path = f"{base_name}_confidence.tif"
            
            confidence_profile = output_profile.copy()
            confidence_profile.update({
                'dtype': 'float32',
                'nodata': -9999.0
            })
            
            with rasterio.open(confidence_path, 'w', **confidence_profile) as dst:
                dst.write(confidence.astype(np.float32), 1)
            
            print(f"✓ Confidence map saved: {confidence_path}")
        
        # Print statistics
        self._print_statistics(final_mask, confidence, output_profile.get('transform'))
        
        # Cleanup
        import shutil
        if not save_patches and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        print(f"\n{'='*70}")
        print("PREDICTION COMPLETED")
        print(f"{'='*70}")
    
    def _save_patch(self, patch_info, profile, output_nodata):
        """Save individual patch"""
        x, y = patch_info['coords']
        probs = patch_info['probs']
        
        if self.mode == 'binary':
            patch_mask = (probs > self.threshold).astype(np.uint8)
            if self.background_value == 0:
                patch_mask = patch_mask  # Already 0/1
            else:
                patch_mask = np.where(patch_mask == 1, 0, 1)  # Invert
        else:
            patch_mask = np.argmax(probs, axis=0).astype(np.uint8)
        
        patch_path = os.path.join(self.temp_dir, f"patch_{y}_{x}.tif")
        patch_profile = profile.copy()
        patch_profile.update({
            'height': patch_mask.shape[0],
            'width': patch_mask.shape[1],
            'transform': rasterio.windows.transform(patch_info['window'], profile['transform']),
            'count': 1,
            'dtype': 'uint8',
            'nodata': output_nodata
        })
        
        with rasterio.open(patch_path, 'w', **patch_profile) as dst:
            dst.write(patch_mask, 1)
    
    def _print_statistics(self, final_mask, confidence, transform):
        """Print prediction statistics"""
        valid_mask = final_mask != 255  # Exclude nodata
        
        if not np.any(valid_mask):
            print("No valid pixels in prediction!")
            return
        
        valid_pixels = final_mask[valid_mask]
        
        print(f"\n{'='*50}")
        print("PREDICTION STATISTICS")
        print(f"{'='*50}")
        
        # Class distribution
        unique, counts = np.unique(valid_pixels, return_counts=True)
        total = counts.sum()
        
        print(f"Class distribution ({self.mode}):")
        for cls, count in zip(unique, counts):
            percentage = (count / total) * 100
            if self.mode == 'binary':
                label = "Foreground" if cls == 1 else "Background"
                print(f"  {label}: {count:,} pixels ({percentage:6.2f}%)")
            else:
                print(f"  Class {cls:2d}: {count:10,} pixels ({percentage:6.2f}%)")
        
        # Confidence statistics
        if confidence is not None and np.any(valid_mask):
            valid_conf = confidence[valid_mask]
            print(f"\nConfidence statistics:")
            print(f"  Average: {np.mean(valid_conf):.4f}")
            print(f"  Min: {np.min(valid_conf):.4f}")
            print(f"  Max: {np.max(valid_conf):.4f}")
            print(f"  Low (<0.5): {np.sum(valid_conf < 0.5):,} pixels")
            print(f"  High (>0.8): {np.sum(valid_conf > 0.8):,} pixels")
        
        # Area calculation if transform available
        if transform and hasattr(transform, '__len__') and len(transform) >= 6:
            pixel_area = abs(transform[0] * transform[4])
            if pixel_area > 0:
                total_area = total * pixel_area
                print(f"\nArea statistics:")
                print(f"  Pixel size: {abs(transform[0]):.2f} x {abs(transform[4]):.2f} units")
                print(f"  Total area: {total_area:,.2f} square units")
    
    def cleanup(self):
        """Cleanup temporary directory"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='Universal large image predictor for segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Binary segmentation with default encoder (resnet34)
  python predict_large_image.py --model model.pth --input image.tif --output prediction.tif
  
  # Specify encoder (must match training encoder!)
  python predict_large_image.py --model model.pth --input image.tif --output prediction.tif --encoder_name resnet50
  
  # Multi-class with 6 classes and EfficientNet encoder
  python predict_large_image.py --model model.pth --input image.tif --output prediction.tif --num_classes 6 --encoder_name efficientnet-b4
  
  # Custom patch size and overlap
  python predict_large_image.py --model model.pth --input image.tif --output prediction.tif --patch_size 256 --overlap 64
  
  # With confidence map
  python predict_large_image.py --model model.pth --input image.tif --output prediction.tif --save_confidence
        '''
    )
    
    # Required arguments
    parser.add_argument('--model', required=True, help='Path to .pth model file')
    parser.add_argument('--input', required=True, help='Path to input image')
    parser.add_argument('--output', required=True, help='Path for output mask')
    
    # Model parameters
    parser.add_argument('--model_name', help='Model architecture name (unet, deeplabv3+, fpn, etc.)')
    parser.add_argument('--encoder_name', 
                        help='''Encoder backbone name from SMP library. Popular choices:
  ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
  EfficientNet: efficientnet-b0 to efficientnet-b7
  ResNeXt: resnext50_32x4d, resnext101_32x8d
  SE-ResNet: se_resnet50, se_resnet101, se_resnet152
  DenseNet: densenet121, densenet169, densenet201
  VGG: vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
  MobileNet: mobilenet_v2, timm-mobilenetv3_large_100
  Full list: https://github.com/qubvel/segmentation_models.pytorch#encoders''')
    parser.add_argument('--in_channels', type=int, help='Number of input channels')
    parser.add_argument('--num_classes', type=int, help='Number of output classes')
    
    # Processing parameters
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size')
    parser.add_argument('--overlap', type=int, default=128, help='Overlap between patches')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary')
    parser.add_argument('--background_value', type=int, default=0, help='Background value')
    parser.add_argument('--normalization_percentile', type=int, default=99, help='Percentile for normalization')
    parser.add_argument('--normalize_per_channel', action='store_true', help='Normalize each channel separately')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device for inference')
    
    # Output options
    parser.add_argument('--save_confidence', action='store_true', help='Save confidence map')
    parser.add_argument('--save_patches', action='store_true', help='Save individual patches')
    parser.add_argument('--output_nodata', type=int, default=255, help='Nodata value for output')
    
    args = parser.parse_args()
    
    # Validate
    if not os.path.exists(args.model):
        print(f"✗ Error: Model not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"✗ Error: Input image not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    try:
        predictor = LargeImagePredictorHybrid(
            model_path=args.model,
            model_name=args.model_name,
            encoder_name=args.encoder_name,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            overlap=args.overlap,
            device=args.device,
            threshold=args.threshold,
            normalization_percentile=args.normalization_percentile,
            background_value=args.background_value,
            normalize_per_channel=args.normalize_per_channel
        )
        
        predictor.predict_large_image(
            input_image_path=args.input,
            output_path=args.output,
            save_confidence=args.save_confidence,
            save_patches=args.save_patches,
            output_nodata=args.output_nodata
        )
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
