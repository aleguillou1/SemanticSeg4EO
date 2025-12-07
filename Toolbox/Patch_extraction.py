# Patch_extraction.py
import os
import random
import argparse
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json


def extract_patches(
    image_path: str,
    label_path: str,
    shapefile_path: str,
    output_dir: str,
    patch_size: int = 224,
    image_channels: int = 4,
    label_channels: int = 1,
    train_ratio: float = 0.75,
    val_ratio: float = 0.2,
    test_ratio: float = 0.05,
    id_column: str = 'AUTO',
    random_seed: Optional[int] = None,
    interpolation: str = 'bilinear',
    compression: bool = True,
    save_metadata: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Extract patches from satellite imagery and labels using a grid shapefile.
    
    Args:
        image_path: Path to input satellite image (GeoTIFF)
        label_path: Path to label image (GeoTIFF)
        shapefile_path: Path to grid shapefile defining patch boundaries
        output_dir: Base output directory
        patch_size: Size of patches (square)
        image_channels: Number of channels in input image
        label_channels: Number of channels in label image
        train_ratio: Ratio of patches for training
        val_ratio: Ratio of patches for validation
        test_ratio: Ratio of patches for testing
        id_column: Column name in shapefile containing patch IDs
        random_seed: Random seed for reproducible splits
        interpolation: Interpolation method ('nearest', 'bilinear', 'bicubic', 'lanczos')
        compression: Enable LZW compression
        save_metadata: Save metadata JSON file
        verbose: Enable verbose output
        
    Returns:
        Dictionary containing extraction statistics and metadata
    """
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Create output directory structure
    splits = ['train', 'validation', 'test']
    subdirs = ['images', 'labels']
    
    dir_paths = {}
    for split in splits:
        dir_paths[split] = {}
        for subdir in subdirs:
            path = os.path.join(output_dir, split, subdir)
            os.makedirs(path, exist_ok=True)
            dir_paths[split][subdir] = path
            if verbose:
                print(f"📁 Created directory: {path}")
    
    # Configure interpolation method
    interpolation_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    interp_method = interpolation_map.get(interpolation.lower(), cv2.INTER_LINEAR)
    
    # Statistics and metadata
    stats = {
        'total_patches': 0,
        'train_patches': 0,
        'val_patches': 0,
        'test_patches': 0,
        'resized_patches': 0,
        'failed_patches': 0
    }
    
    metadata = []
    
    # Open raster files
    with rasterio.open(image_path) as src_img, rasterio.open(label_path) as src_lbl:
        
        # Read grid shapefile
        if verbose:
            print(f"📋 Loading grid from: {shapefile_path}")
        grid = gpd.read_file(shapefile_path)
        stats['total_patches'] = len(grid)
        
        # Validate ID column exists
        if id_column not in grid.columns:
            raise ValueError(f"ID column '{id_column}' not found in shapefile")
        
        if verbose:
            print(f"📊 Grid contains {len(grid)} patches")
            print(f"📐 Patch size: {patch_size}x{patch_size}")
            print(f"🎨 Image channels: {image_channels}, Label channels: {label_channels}")
        
        # Shuffle indices for random distribution
        indices = list(grid.index)
        random.shuffle(indices)
        
        # Calculate split sizes
        num_patches = len(indices)
        num_train = int(train_ratio * num_patches)
        num_val = int(val_ratio * num_patches)
        num_test = num_patches - num_train - num_val
        
        # Split indices
        train_indices = set(indices[:num_train])
        val_indices = set(indices[num_train:num_train + num_val])
        test_indices = set(indices[num_train + num_val:])
        
        stats.update({
            'train_patches': len(train_indices),
            'val_patches': len(val_indices),
            'test_patches': len(test_indices)
        })
        
        if verbose:
            print(f"📈 Split distribution:")
            print(f"   - Training: {len(train_indices)} patches ({train_ratio*100:.1f}%)")
            print(f"   - Validation: {len(val_indices)} patches ({val_ratio*100:.1f}%)")
            print(f"   - Testing: {len(test_indices)} patches ({test_ratio*100:.1f}%)")
        
        # Process each patch
        for idx, row in grid.iterrows():
            geometry = [row['geometry']]
            patch_id = row[id_column]
            
            try:
                # Extract image and label patches
                patch_img, transform = mask(src_img, geometry, crop=True)
                patch_lbl, _ = mask(src_lbl, geometry, crop=True)
                
                # Get current dimensions
                img_height, img_width = patch_img.shape[1], patch_img.shape[2]
                lbl_height, lbl_width = patch_lbl.shape[1], patch_lbl.shape[2]
                
                # Determine if resizing is needed
                needs_resize = (img_height != patch_size or img_width != patch_size or 
                               lbl_height != patch_size or lbl_width != patch_size)
                
                if needs_resize:
                    stats['resized_patches'] += 1
                    
                    # Resize image patch
                    if img_height != patch_size or img_width != patch_size:
                        resized_patch_img = np.zeros((image_channels, patch_size, patch_size), 
                                                     dtype=patch_img.dtype)
                        # Process each channel
                        for i in range(min(image_channels, patch_img.shape[0])):
                            resized_patch_img[i] = cv2.resize(
                                patch_img[i], 
                                (patch_size, patch_size), 
                                interpolation=interp_method
                            )
                        patch_img = resized_patch_img
                    
                    # Resize label patch
                    if lbl_height != patch_size or lbl_width != patch_size:
                        if label_channels == 1:
                            # Single channel label
                            resized_patch_lbl = cv2.resize(
                                patch_lbl[0], 
                                (patch_size, patch_size), 
                                interpolation=cv2.INTER_NEAREST
                            )
                            patch_lbl = resized_patch_lbl[np.newaxis, :, :]
                        else:
                            # Multi-channel label
                            resized_patch_lbl = np.zeros((label_channels, patch_size, patch_size), 
                                                         dtype=patch_lbl.dtype)
                            for i in range(label_channels):
                                resized_patch_lbl[i] = cv2.resize(
                                    patch_lbl[i], 
                                    (patch_size, patch_size), 
                                    interpolation=cv2.INTER_NEAREST
                                )
                            patch_lbl = resized_patch_lbl
                
                # Determine output split
                if idx in train_indices:
                    split = 'train'
                elif idx in val_indices:
                    split = 'validation'
                elif idx in test_indices:
                    split = 'test'
                else:
                    continue  # Should not happen
                
                # Create filename
                patch_filename = f"patch_{patch_id}.tif"
                
                # Configure compression
                create_options = {}
                if compression:
                    create_options['compress'] = 'lzw'
                
                # Save image patch
                img_output_path = os.path.join(dir_paths[split]['images'], patch_filename)
                with rasterio.open(
                    img_output_path, 'w',
                    driver='GTiff',
                    height=patch_size,
                    width=patch_size,
                    count=image_channels,
                    dtype=patch_img.dtype,
                    crs=src_img.crs,
                    transform=transform,
                    **create_options
                ) as dst_img:
                    dst_img.write(patch_img[:image_channels])
                
                # Save label patch
                lbl_output_path = os.path.join(dir_paths[split]['labels'], patch_filename)
                with rasterio.open(
                    lbl_output_path, 'w',
                    driver='GTiff',
                    height=patch_size,
                    width=patch_size,
                    count=label_channels,
                    dtype=patch_lbl.dtype,
                    crs=src_lbl.crs,
                    transform=transform,
                    **create_options
                ) as dst_lbl:
                    dst_lbl.write(patch_lbl[:label_channels])
                
                # Record metadata
                metadata.append({
                    'patch_id': int(patch_id) if isinstance(patch_id, (int, np.integer)) else str(patch_id),
                    'split': split,
                    'image_path': img_output_path,
                    'label_path': lbl_output_path,
                    'original_size': {'height': img_height, 'width': img_width},
                    'resized': needs_resize,
                    'transform': {
                        'a': transform.a,
                        'b': transform.b,
                        'c': transform.c,
                        'd': transform.d,
                        'e': transform.e,
                        'f': transform.f
                    }
                })
                
                if verbose and len(metadata) % 100 == 0:
                    print(f"   Processed {len(metadata)}/{num_patches} patches...")
                
            except Exception as e:
                stats['failed_patches'] += 1
                print(f"⚠️  Error processing patch {patch_id}: {str(e)[:100]}")
                continue
        
        # Save metadata if requested
        if save_metadata and metadata:
            metadata_path = os.path.join(output_dir, "patch_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({
                    'statistics': stats,
                    'patches': metadata,
                    'config': {
                        'patch_size': patch_size,
                        'image_channels': image_channels,
                        'label_channels': label_channels,
                        'train_ratio': train_ratio,
                        'val_ratio': val_ratio,
                        'test_ratio': test_ratio,
                        'random_seed': random_seed,
                        'interpolation': interpolation
                    }
                }, f, indent=2)
            if verbose:
                print(f"📄 Metadata saved: {metadata_path}")
    
    return {'statistics': stats, 'metadata': metadata}


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalize image by dividing by max value"""
    max_val = np.max(img)
    if max_val > 0:
        return img / max_val
    return img


def load_single_patch(image_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single image patch and its corresponding label"""
    with rasterio.open(image_path) as src:
        img = src.read()  # Shape: (C, H, W)
        img = img.transpose(1, 2, 0).astype(np.float32)  # Convert to (H, W, C)
    
    with rasterio.open(label_path) as src:
        mask = src.read()  # Shape: (C, H, W)
        if mask.shape[0] == 1:
            mask = mask[0]  # Convert to (H, W)
        else:
            mask = mask.transpose(1, 2, 0)  # Convert to (H, W, C)
    
    return img, mask


def visualize_sample(
    output_dir: str,
    split: str = 'train',
    sample_index: int = 0,
    image_channels: int = 4,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Visualize a sample patch from the extracted dataset.
    
    Args:
        output_dir: Base output directory
        split: Which split to visualize ('train', 'validation', 'test')
        sample_index: Index of the sample to visualize
        image_channels: Number of image channels (for display)
        figsize: Figure size for matplotlib
    """
    
    # Find image files in the split directory
    image_dir = os.path.join(output_dir, split, 'images')
    label_dir = os.path.join(output_dir, split, 'labels')
    
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"❌ Directory not found: {image_dir} or {label_dir}")
        return
    
    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.tif', '.tiff'))])
    
    if not image_files:
        print(f"❌ No image files found in {image_dir}")
        return
    
    # Check if sample_index is valid
    if sample_index >= len(image_files):
        print(f"⚠️  Sample index {sample_index} out of range. Using first sample.")
        sample_index = 0
    
    # Load the sample
    image_file = image_files[sample_index]
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, image_file)
    
    if not os.path.exists(label_path):
        print(f"❌ Corresponding label not found: {label_path}")
        return
    
    print(f"📷 Loading sample: {image_file}")
    img, mask = load_single_patch(image_path, label_path)
    
    # Normalize image for visualization
    img_normalized = normalize_img(img)
    
    # Create visualization
    plt.figure(figsize=figsize)
    
    # Display image (show first 3 channels as RGB if available)
    plt.subplot(1, 3, 1)
    if img_normalized.shape[-1] >= 3:
        # Show RGB (first 3 channels)
        plt.imshow(img_normalized[:, :, :3])
        plt.title(f'Image (RGB channels)\nShape: {img.shape}')
    elif img_normalized.shape[-1] == 1:
        # Grayscale
        plt.imshow(img_normalized[:, :, 0], cmap='gray')
        plt.title(f'Image (Single channel)\nShape: {img.shape}')
    else:
        # Show first channel
        plt.imshow(img_normalized[:, :, 0], cmap='viridis')
        plt.title(f'Image (Channel 0)\nShape: {img.shape}')
    
    # Display all channels separately if <= 4 channels
    if img_normalized.shape[-1] <= 4 and img_normalized.shape[-1] > 1:
        plt.subplot(1, 3, 2)
        channels_display = min(4, img_normalized.shape[-1])
        for i in range(channels_display):
            plt.subplot(2, 2, i + 3)
            plt.imshow(img_normalized[:, :, i], cmap='viridis')
            plt.title(f'Channel {i}')
            plt.axis('off')
    else:
        plt.subplot(1, 3, 2)
        if len(mask.shape) == 2:
            plt.imshow(mask, cmap='gray')
            plt.title(f'Mask\nShape: {mask.shape}')
            plt.colorbar()
        else:
            plt.imshow(mask[:, :, 0], cmap='gray')
            plt.title(f'Mask (Channel 0)\nShape: {mask.shape}')
            plt.colorbar()
    
    # Display mask
    plt.subplot(1, 3, 3)
    if len(mask.shape) == 2:
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask\nUnique values: {np.unique(mask)}')
        plt.colorbar()
    else:
        if mask.shape[-1] == 1:
            plt.imshow(mask[:, :, 0], cmap='gray')
            plt.title(f'Mask\nUnique values: {np.unique(mask[:, :, 0])}')
            plt.colorbar()
        else:
            # For multi-channel masks, show first channel
            plt.imshow(mask[:, :, 0], cmap='viridis')
            plt.title(f'Mask (Channel 0)\nShape: {mask.shape}')
            plt.colorbar()
    
    plt.tight_layout()
    plt.suptitle(f'Sample from {split} set: {image_file}', y=1.02, fontsize=14)
    plt.show()
    
    # Print statistics
    print(f"\n📊 Sample Statistics:")
    print(f"   - Image shape: {img.shape}")
    print(f"   - Image dtype: {img.dtype}")
    print(f"   - Image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"   - Mask shape: {mask.shape}")
    print(f"   - Mask dtype: {mask.dtype}")
    print(f"   - Mask unique values: {np.unique(mask)}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract patches from satellite imagery and labels using a grid shapefile',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction with defaults
  python extract_patches_cli.py extract \\
    --image path/to/image.tif \\
    --label path/to/label.tif \\
    --grid path/to/grid.shp \\
    --output path/to/output
  
  # Custom parameters
  python extract_patches_cli.py extract \\
    --image image.tif \\
    --label label.tif \\
    --grid grid.shp \\
    --output patches_256 \\
    --patch_size 256 \\
    --image_channels 10 \\
    --train_ratio 0.8 \\
    --val_ratio 0.15 \\
    --test_ratio 0.05 \\
    --random_seed 42 \\
    --interpolation bicubic \\
    --save_metadata
  
  # Visualize extracted patches
  python extract_patches_cli.py visualize \\
    --output path/to/output \\
    --split validation \\
    --sample_index 5
        """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract patches from imagery')
    
    # Required arguments for extraction
    extract_parser.add_argument('--image', required=True, help='Path to input satellite image (GeoTIFF)')
    extract_parser.add_argument('--label', required=True, help='Path to label image (GeoTIFF)')
    extract_parser.add_argument('--grid', required=True, help='Path to grid shapefile')
    extract_parser.add_argument('--output', required=True, help='Base output directory')
    
    # Patch parameters
    extract_parser.add_argument('--patch_size', type=int, default=224, help='Size of patches (default: 224)')
    extract_parser.add_argument('--image_channels', type=int, default=4, help='Number of image channels (default: 4)')
    extract_parser.add_argument('--label_channels', type=int, default=1, help='Number of label channels (default: 1)')
    
    # Split parameters
    extract_parser.add_argument('--train_ratio', type=float, default=0.75, help='Training ratio (default: 0.75)')
    extract_parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio (default: 0.2)')
    extract_parser.add_argument('--test_ratio', type=float, default=0.05, help='Test ratio (default: 0.05)')
    
    # Advanced parameters
    extract_parser.add_argument('--id_column', default='AUTO', help='Patch ID column (default: AUTO)')
    extract_parser.add_argument('--random_seed', type=int, help='Random seed for reproducible splits')
    extract_parser.add_argument('--interpolation', default='bilinear', 
                        choices=['nearest', 'bilinear', 'bicubic', 'lanczos'],
                        help='Interpolation method (default: bilinear)')
    extract_parser.add_argument('--no_compression', action='store_true', help='Disable LZW compression')
    extract_parser.add_argument('--save_metadata', action='store_true', help='Save metadata JSON file')
    extract_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Visualize extracted patches')
    visualize_parser.add_argument('--output', required=True, help='Output directory from extraction')
    visualize_parser.add_argument('--split', default='train', choices=['train', 'validation', 'test'],
                                  help='Dataset split to visualize (default: train)')
    visualize_parser.add_argument('--sample_index', type=int, default=0, help='Sample index to visualize (default: 0)')
    visualize_parser.add_argument('--image_channels', type=int, default=4, help='Number of image channels (for display)')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        # Validate ratios
        total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            print(f"❌ Error: Ratios must sum to 1.0 (got {total_ratio:.3f})")
            print(f"   Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}")
            return 1
        
        # Validate patch size
        if args.patch_size <= 0:
            print(f"❌ Error: Patch size must be positive (got {args.patch_size})")
            return 1
        
        # Validate channels
        if args.image_channels <= 0:
            print(f"❌ Error: Image channels must be positive (got {args.image_channels})")
            return 1
        
        if args.label_channels <= 0:
            print(f"❌ Error: Label channels must be positive (got {args.label_channels})")
            return 1
        
        print("🚀 Starting patch extraction...")
        print(f"📁 Output directory: {args.output}")
        
        try:
            results = extract_patches(
                image_path=args.image,
                label_path=args.label,
                shapefile_path=args.grid,
                output_dir=args.output,
                patch_size=args.patch_size,
                image_channels=args.image_channels,
                label_channels=args.label_channels,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                id_column=args.id_column,
                random_seed=args.random_seed,
                interpolation=args.interpolation,
                compression=not args.no_compression,
                save_metadata=args.save_metadata,
                verbose=args.verbose
            )
            
            stats = results['statistics']
            print(f"\n✅ Extraction completed successfully!")
            print(f"\n📊 Extraction Statistics:")
            print(f"   - Total patches processed: {stats['total_patches']}")
            print(f"   - Training patches: {stats['train_patches']}")
            print(f"   - Validation patches: {stats['val_patches']}")
            print(f"   - Test patches: {stats['test_patches']}")
            print(f"   - Resized patches: {stats['resized_patches']}")
            print(f"   - Failed patches: {stats['failed_patches']}")
            
            # Show directory structure
            print(f"\n📁 Output structure created at: {args.output}")
            for root, dirs, files in os.walk(args.output):
                level = root.replace(args.output, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files
                    if file.endswith('.tif'):
                        print(f'{subindent}{file}')
                if len(files) > 5:
                    print(f'{subindent}... and {len(files) - 5} more files')
            
            print(f"\n💡 Tip: Use 'visualize' command to view extracted patches:")
            print(f"     python extract_patches_cli.py visualize --output {args.output}")
            
        except Exception as e:
            print(f"\n❌ Error during extraction: {e}")
            return 1
    
    elif args.command == 'visualize':
        print(f"🎨 Visualizing patches from {args.output}")
        try:
            visualize_sample(
                output_dir=args.output,
                split=args.split,
                sample_index=args.sample_index,
                image_channels=args.image_channels
            )
        except Exception as e:
            print(f"❌ Error during visualization: {e}")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())