# Patch_extraction_v2.py
"""
Patch Extraction Tool - Version 2.1
Supports both single image/label mode and batch processing mode.

Batch mode naming convention:
    - Images: Image_1.tif, Image_2.tif, ... OR image_1.tif, image_2.tif, ...
    - Labels: Label_1.tif, Label_2.tif, ... OR label_1.tif, label_2.tif, ...
    - Grids (optional per-image): Grid_1.shp, Grid_2.shp, ... OR use single grid for all
"""

import os
import re
import sys
import random
import argparse
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import json
from dataclasses import dataclass
from collections import defaultdict

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("💡 Tip: Install tqdm for progress bar: pip install tqdm")


@dataclass
class ImageLabelPair:
    """Represents a matched image-label pair with optional grid"""
    pair_id: str
    image_path: str
    label_path: str
    grid_path: Optional[str] = None
    
    def __repr__(self):
        return f"Pair({self.pair_id}: {Path(self.image_path).name} <-> {Path(self.label_path).name})"


def find_matching_pairs(
    data_dir: str,
    image_pattern: str = r'^[Ii]mage[_-]?(\d+)\.tif{1,2}$',
    label_pattern: str = r'^[Ll]abel[_-]?(\d+)\.tif{1,2}$',
    grid_pattern: str = r'^[Gg]rid[_-]?(\d+)\.shp$',
    recursive: bool = True,
    verbose: bool = False
) -> Tuple[List[ImageLabelPair], Dict]:
    """
    Find matching image-label pairs in a directory.
    
    Args:
        data_dir: Directory to search for files
        image_pattern: Regex pattern for image files (must have group for ID)
        label_pattern: Regex pattern for label files (must have group for ID)
        grid_pattern: Regex pattern for grid files (optional, must have group for ID)
        recursive: Search subdirectories recursively
        verbose: Print detailed matching information
    
    Returns:
        Tuple of (list of matched pairs, stats dictionary)
    """
    images = {}  # id -> path
    labels = {}  # id -> path
    grids = {}   # id -> path
    
    # Compile patterns
    img_re = re.compile(image_pattern)
    lbl_re = re.compile(label_pattern)
    grd_re = re.compile(grid_pattern)
    
    # Walk through directory
    if recursive:
        walker = os.walk(data_dir)
    else:
        walker = [(data_dir, [], os.listdir(data_dir))]
    
    for root, dirs, files in walker:
        for filename in files:
            filepath = os.path.join(root, filename)
            
            # Check for image match
            img_match = img_re.match(filename)
            if img_match:
                pair_id = img_match.group(1)
                images[pair_id] = filepath
                if verbose:
                    print(f"  📷 Found image: {filename} (ID: {pair_id})")
                continue
            
            # Check for label match
            lbl_match = lbl_re.match(filename)
            if lbl_match:
                pair_id = lbl_match.group(1)
                labels[pair_id] = filepath
                if verbose:
                    print(f"  🏷️  Found label: {filename} (ID: {pair_id})")
                continue
            
            # Check for grid match
            grd_match = grd_re.match(filename)
            if grd_match:
                pair_id = grd_match.group(1)
                grids[pair_id] = filepath
                if verbose:
                    print(f"  📐 Found grid: {filename} (ID: {pair_id})")
    
    # Match pairs
    pairs = []
    all_ids = set(images.keys()) | set(labels.keys())
    matched_ids = set(images.keys()) & set(labels.keys())
    unmatched_images = set(images.keys()) - matched_ids
    unmatched_labels = set(labels.keys()) - matched_ids
    
    for pair_id in sorted(matched_ids, key=lambda x: int(x) if x.isdigit() else x):
        pair = ImageLabelPair(
            pair_id=pair_id,
            image_path=images[pair_id],
            label_path=labels[pair_id],
            grid_path=grids.get(pair_id)  # Optional per-pair grid
        )
        pairs.append(pair)
    
    stats = {
        'total_images': len(images),
        'total_labels': len(labels),
        'total_grids': len(grids),
        'matched_pairs': len(pairs),
        'unmatched_images': list(unmatched_images),
        'unmatched_labels': list(unmatched_labels)
    }
    
    return pairs, stats


def extract_patches_single(
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
    verbose: bool = False,
    prefix: str = "",
    existing_splits: Optional[Dict] = None
) -> Dict:
    """
    Extract patches from a single satellite image and label using a grid shapefile.
    
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
        prefix: Prefix for output filenames (useful for batch mode)
        existing_splits: Pre-computed split indices (for consistent splits in batch mode)
        
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
            print(f"  📋 Loading grid from: {shapefile_path}")
        grid = gpd.read_file(shapefile_path)
        stats['total_patches'] = len(grid)
        
        # Validate ID column exists
        if id_column not in grid.columns:
            raise ValueError(f"ID column '{id_column}' not found in shapefile")
        
        print(f"  📊 Grid contains {len(grid)} patches", flush=True)
        
        # Handle splits
        indices = list(grid.index)
        num_patches = len(indices)
        
        if existing_splits is None:
            # Shuffle indices for random distribution
            random.shuffle(indices)
            
            # Calculate split sizes
            num_train = int(train_ratio * num_patches)
            num_val = int(val_ratio * num_patches)
            
            # Split indices
            train_indices = set(indices[:num_train])
            val_indices = set(indices[num_train:num_train + num_val])
            test_indices = set(indices[num_train + num_val:])
        else:
            train_indices = existing_splits['train']
            val_indices = existing_splits['validation']
            test_indices = existing_splits['test']
        
        stats.update({
            'train_patches': len([i for i in indices if i in train_indices]),
            'val_patches': len([i for i in indices if i in val_indices]),
            'test_patches': len([i for i in indices if i in test_indices])
        })
        
        # Print split distribution
        print(f"  📈 Split distribution:", flush=True)
        print(f"     - Training: {stats['train_patches']} patches ({train_ratio*100:.1f}%)", flush=True)
        print(f"     - Validation: {stats['val_patches']} patches ({val_ratio*100:.1f}%)", flush=True)
        print(f"     - Testing: {stats['test_patches']} patches ({test_ratio*100:.1f}%)", flush=True)
        print(f"  🚀 Processing patches...", flush=True)
        
        # Setup progress tracking
        if HAS_TQDM:
            iterator = tqdm(grid.iterrows(), total=num_patches, desc="  Extracting", unit="patch")
        else:
            iterator = grid.iterrows()
        
        processed = 0
        
        # Process each patch
        for idx, row in iterator:
            geometry = [row['geometry']]
            patch_id = row[id_column]
            
            # Progress update every 100 patches (works with or without tqdm)
            processed += 1
            if processed % 100 == 0:
                if HAS_TQDM:
                    tqdm.write(f"     ✓ Processed {processed}/{num_patches} patches...")
                else:
                    print(f"     Processed {processed}/{num_patches} patches...", flush=True)
            
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
                            resized_patch_lbl = cv2.resize(
                                patch_lbl[0], 
                                (patch_size, patch_size), 
                                interpolation=cv2.INTER_NEAREST
                            )
                            patch_lbl = resized_patch_lbl[np.newaxis, :, :]
                        else:
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
                    continue
                
                # Create filename with optional prefix
                if prefix:
                    patch_filename = f"{prefix}_patch_{patch_id}.tif"
                else:
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
                    'source_prefix': prefix if prefix else "default",
                    'split': split,
                    'image_path': img_output_path,
                    'label_path': lbl_output_path,
                    'original_size': {'height': img_height, 'width': img_width},
                    'resized': needs_resize,
                    'transform': {
                        'a': transform.a, 'b': transform.b, 'c': transform.c,
                        'd': transform.d, 'e': transform.e, 'f': transform.f
                    }
                })
                
            except Exception as e:
                stats['failed_patches'] += 1
                if verbose:
                    print(f"  ⚠️  Error processing patch {patch_id}: {str(e)[:100]}", flush=True)
                continue
    
    return {'statistics': stats, 'metadata': metadata}


def extract_patches_batch(
    data_dir: str,
    grid_path: str,
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
    save_metadata: bool = True,
    recursive: bool = True,
    image_pattern: str = r'^[Ii]mage[_-]?(\d+)\.tif{1,2}$',
    label_pattern: str = r'^[Ll]abel[_-]?(\d+)\.tif{1,2}$',
    use_per_image_grid: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Extract patches from multiple image-label pairs in batch mode.
    
    Args:
        data_dir: Directory containing image-label pairs
        grid_path: Path to grid shapefile (used for all images if use_per_image_grid=False)
        output_dir: Base output directory
        patch_size: Size of patches (square)
        image_channels: Number of channels in input images
        label_channels: Number of channels in label images
        train_ratio: Ratio of patches for training
        val_ratio: Ratio of patches for validation
        test_ratio: Ratio of patches for testing
        id_column: Column name in shapefile containing patch IDs
        random_seed: Random seed for reproducible splits
        interpolation: Interpolation method
        compression: Enable LZW compression
        save_metadata: Save metadata JSON file
        recursive: Search data_dir recursively
        image_pattern: Regex pattern for matching image files
        label_pattern: Regex pattern for matching label files
        use_per_image_grid: Use Grid_X.shp for each Image_X.tif instead of single grid
        verbose: Enable verbose output
        
    Returns:
        Dictionary containing aggregated statistics and metadata
    """
    
    print(f"\n🔍 Scanning directory: {data_dir}")
    print(f"   Recursive search: {'Yes' if recursive else 'No'}")
    
    # Find matching pairs
    pairs, scan_stats = find_matching_pairs(
        data_dir=data_dir,
        image_pattern=image_pattern,
        label_pattern=label_pattern,
        recursive=recursive,
        verbose=verbose
    )
    
    print(f"\n📊 Scan Results:")
    print(f"   - Images found: {scan_stats['total_images']}")
    print(f"   - Labels found: {scan_stats['total_labels']}")
    print(f"   - Matched pairs: {scan_stats['matched_pairs']}")
    
    if scan_stats['unmatched_images']:
        print(f"   ⚠️  Unmatched images (IDs): {scan_stats['unmatched_images']}")
    if scan_stats['unmatched_labels']:
        print(f"   ⚠️  Unmatched labels (IDs): {scan_stats['unmatched_labels']}")
    
    if not pairs:
        print("\n❌ No matching image-label pairs found!")
        print("   Make sure files follow naming convention:")
        print("   - Images: Image_1.tif, Image_2.tif, ...")
        print("   - Labels: Label_1.tif, Label_2.tif, ...")
        return {'statistics': {}, 'metadata': [], 'pairs_processed': 0}
    
    print(f"\n📋 Found {len(pairs)} pairs to process:")
    for pair in pairs:
        print(f"   - {pair}")
    
    # Aggregated statistics
    total_stats = {
        'pairs_processed': 0,
        'total_patches': 0,
        'train_patches': 0,
        'val_patches': 0,
        'test_patches': 0,
        'resized_patches': 0,
        'failed_patches': 0,
        'per_pair_stats': {}
    }
    all_metadata = []
    
    # Process each pair
    for i, pair in enumerate(pairs, 1):
        print(f"\n{'='*60}")
        print(f"📷 Processing pair {i}/{len(pairs)}: {pair.pair_id}")
        print(f"   Image: {Path(pair.image_path).name}")
        print(f"   Label: {Path(pair.label_path).name}")
        
        # Determine grid path
        if use_per_image_grid and pair.grid_path:
            current_grid = pair.grid_path
            print(f"   Grid: {Path(current_grid).name} (per-image)")
        else:
            current_grid = grid_path
            print(f"   Grid: {Path(current_grid).name} (shared)")
        
        try:
            # Extract patches for this pair
            result = extract_patches_single(
                image_path=pair.image_path,
                label_path=pair.label_path,
                shapefile_path=current_grid,
                output_dir=output_dir,
                patch_size=patch_size,
                image_channels=image_channels,
                label_channels=label_channels,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                id_column=id_column,
                random_seed=random_seed + i if random_seed else None,  # Different seed per pair
                interpolation=interpolation,
                compression=compression,
                save_metadata=False,  # We'll save aggregated metadata later
                verbose=verbose,
                prefix=f"img{pair.pair_id}"  # Add prefix to distinguish sources
            )
            
            pair_stats = result['statistics']
            total_stats['pairs_processed'] += 1
            total_stats['total_patches'] += pair_stats['total_patches']
            total_stats['train_patches'] += pair_stats['train_patches']
            total_stats['val_patches'] += pair_stats['val_patches']
            total_stats['test_patches'] += pair_stats['test_patches']
            total_stats['resized_patches'] += pair_stats['resized_patches']
            total_stats['failed_patches'] += pair_stats['failed_patches']
            total_stats['per_pair_stats'][pair.pair_id] = pair_stats
            
            all_metadata.extend(result['metadata'])
            
            print(f"   ✅ Extracted {pair_stats['total_patches']} patches")
            print(f"      Train: {pair_stats['train_patches']}, Val: {pair_stats['val_patches']}, Test: {pair_stats['test_patches']}")
            
        except Exception as e:
            print(f"   ❌ Error processing pair: {e}")
            total_stats['per_pair_stats'][pair.pair_id] = {'error': str(e)}
    
    # Save aggregated metadata
    if save_metadata and all_metadata:
        metadata_path = os.path.join(output_dir, "patch_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                'mode': 'batch',
                'statistics': total_stats,
                'scan_stats': scan_stats,
                'patches': all_metadata,
                'config': {
                    'data_dir': data_dir,
                    'patch_size': patch_size,
                    'image_channels': image_channels,
                    'label_channels': label_channels,
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'test_ratio': test_ratio,
                    'random_seed': random_seed,
                    'interpolation': interpolation,
                    'image_pattern': image_pattern,
                    'label_pattern': label_pattern
                }
            }, f, indent=2)
        print(f"\n📄 Metadata saved: {metadata_path}")
    
    return {'statistics': total_stats, 'metadata': all_metadata}


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalize image by dividing by max value"""
    max_val = np.max(img)
    if max_val > 0:
        return img / max_val
    return img


def load_single_patch(image_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single image patch and its corresponding label"""
    with rasterio.open(image_path) as src:
        img = src.read()
        img = img.transpose(1, 2, 0).astype(np.float32)
    
    with rasterio.open(label_path) as src:
        mask_arr = src.read()
        if mask_arr.shape[0] == 1:
            mask_arr = mask_arr[0]
        else:
            mask_arr = mask_arr.transpose(1, 2, 0)
    
    return img, mask_arr


def visualize_sample(
    output_dir: str,
    split: str = 'train',
    sample_index: int = 0,
    image_channels: int = 4,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Visualize a sample patch from the extracted dataset.
    """
    image_dir = os.path.join(output_dir, split, 'images')
    label_dir = os.path.join(output_dir, split, 'labels')
    
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"❌ Directory not found: {image_dir} or {label_dir}")
        return
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.tif', '.tiff'))])
    
    if not image_files:
        print(f"❌ No image files found in {image_dir}")
        return
    
    if sample_index >= len(image_files):
        print(f"⚠️  Sample index {sample_index} out of range. Using first sample.")
        sample_index = 0
    
    image_file = image_files[sample_index]
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, image_file)
    
    if not os.path.exists(label_path):
        print(f"❌ Corresponding label not found: {label_path}")
        return
    
    print(f"📷 Loading sample: {image_file}")
    img, mask_arr = load_single_patch(image_path, label_path)
    
    img_normalized = normalize_img(img)
    
    plt.figure(figsize=figsize)
    
    # Display image
    plt.subplot(1, 3, 1)
    if img_normalized.shape[-1] >= 3:
        plt.imshow(img_normalized[:, :, :3])
        plt.title(f'Image (RGB channels)\nShape: {img.shape}')
    elif img_normalized.shape[-1] == 1:
        plt.imshow(img_normalized[:, :, 0], cmap='gray')
        plt.title(f'Image (Single channel)\nShape: {img.shape}')
    else:
        plt.imshow(img_normalized[:, :, 0], cmap='viridis')
        plt.title(f'Image (Channel 0)\nShape: {img.shape}')
    plt.axis('off')
    
    # Display mask
    plt.subplot(1, 3, 2)
    if len(mask_arr.shape) == 2:
        plt.imshow(mask_arr, cmap='viridis')
        plt.title(f'Label\nShape: {mask_arr.shape}')
    else:
        plt.imshow(mask_arr[:, :, 0], cmap='viridis')
        plt.title(f'Label (Channel 0)\nShape: {mask_arr.shape}')
    plt.colorbar()
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    if img_normalized.shape[-1] >= 3:
        plt.imshow(img_normalized[:, :, :3])
    else:
        plt.imshow(img_normalized[:, :, 0], cmap='gray')
    if len(mask_arr.shape) == 2:
        plt.imshow(mask_arr, cmap='jet', alpha=0.4)
    else:
        plt.imshow(mask_arr[:, :, 0], cmap='jet', alpha=0.4)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Sample from {split} set: {image_file}', y=1.02, fontsize=14)
    plt.savefig(os.path.join(output_dir, f'sample_{split}_{sample_index}.png'), 
                dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_dir}/sample_{split}_{sample_index}.png")


def list_dataset_info(output_dir: str) -> None:
    """List information about an extracted dataset"""
    print(f"\n📂 Dataset Info: {output_dir}\n")
    
    # Check for metadata
    metadata_path = os.path.join(output_dir, "patch_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        
        print(f"Mode: {meta.get('mode', 'single')}")
        stats = meta.get('statistics', {})
        print(f"\n📊 Statistics:")
        print(f"   Total patches: {stats.get('total_patches', 'N/A')}")
        print(f"   Train: {stats.get('train_patches', 'N/A')}")
        print(f"   Validation: {stats.get('val_patches', 'N/A')}")
        print(f"   Test: {stats.get('test_patches', 'N/A')}")
        
        if 'pairs_processed' in stats:
            print(f"   Pairs processed: {stats['pairs_processed']}")
    
    # Count files
    for split in ['train', 'validation', 'test']:
        img_dir = os.path.join(output_dir, split, 'images')
        lbl_dir = os.path.join(output_dir, split, 'labels')
        
        if os.path.exists(img_dir):
            img_count = len([f for f in os.listdir(img_dir) if f.endswith(('.tif', '.tiff'))])
            lbl_count = len([f for f in os.listdir(lbl_dir) if f.endswith(('.tif', '.tiff'))])
            print(f"\n   {split.capitalize()}:")
            print(f"      Images: {img_count}")
            print(f"      Labels: {lbl_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract patches from satellite imagery - Supports single and batch modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              USAGE EXAMPLES                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  SINGLE MODE (one image + one label):                                        ║
║  ------------------------------------                                        ║
║  python Patch_extraction_v2.py single \\                                     ║
║      --image path/to/image.tif \\                                            ║
║      --label path/to/label.tif \\                                            ║
║      --grid path/to/grid.shp \\                                              ║
║      --output path/to/output                                                 ║
║                                                                              ║
║  BATCH MODE (multiple images + labels):                                      ║
║  --------------------------------------                                      ║
║  Directory structure expected:                                               ║
║    data_folder/                                                              ║
║    ├── Image_1.tif                                                           ║
║    ├── Label_1.tif                                                           ║
║    ├── Image_2.tif                                                           ║
║    ├── Label_2.tif                                                           ║
║    └── ...                                                                   ║
║                                                                              ║
║  python Patch_extraction_v2.py batch \\                                      ║
║      --data_dir path/to/data_folder \\                                       ║
║      --grid path/to/grid.shp \\                                              ║
║      --output path/to/output \\                                              ║
║      --recursive                                                             ║
║                                                                              ║
║  CUSTOM PATTERNS:                                                            ║
║  python Patch_extraction_v2.py batch \\                                      ║
║      --data_dir ./data \\                                                    ║
║      --grid grid.shp \\                                                      ║
║      --output ./patches \\                                                   ║
║      --image_pattern "^sentinel_(\d+)\.tif$" \\                              ║
║      --label_pattern "^mask_(\d+)\.tif$"                                     ║
║                                                                              ║
║  VISUALIZE:                                                                  ║
║  python Patch_extraction_v2.py visualize --output ./patches --split train    ║
║                                                                              ║
║  INFO:                                                                       ║
║  python Patch_extraction_v2.py info --output ./patches                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # ========================
    # SINGLE MODE
    # ========================
    single_parser = subparsers.add_parser('single', help='Extract patches from a single image-label pair')
    single_parser.add_argument('--image', required=True, help='Path to input satellite image (GeoTIFF)')
    single_parser.add_argument('--label', required=True, help='Path to label image (GeoTIFF)')
    single_parser.add_argument('--grid', required=True, help='Path to grid shapefile')
    single_parser.add_argument('--output', required=True, help='Base output directory')
    single_parser.add_argument('--patch_size', type=int, default=224, help='Size of patches (default: 224)')
    single_parser.add_argument('--image_channels', type=int, default=4, help='Number of image channels (default: 4)')
    single_parser.add_argument('--label_channels', type=int, default=1, help='Number of label channels (default: 1)')
    single_parser.add_argument('--train_ratio', type=float, default=0.75, help='Training ratio (default: 0.75)')
    single_parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio (default: 0.2)')
    single_parser.add_argument('--test_ratio', type=float, default=0.05, help='Test ratio (default: 0.05)')
    single_parser.add_argument('--id_column', default='AUTO', help='Patch ID column (default: AUTO)')
    single_parser.add_argument('--random_seed', type=int, help='Random seed for reproducible splits')
    single_parser.add_argument('--interpolation', default='bilinear', 
                               choices=['nearest', 'bilinear', 'bicubic', 'lanczos'],
                               help='Interpolation method (default: bilinear)')
    single_parser.add_argument('--no_compression', action='store_true', help='Disable LZW compression')
    single_parser.add_argument('--save_metadata', action='store_true', help='Save metadata JSON file')
    single_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # ========================
    # BATCH MODE
    # ========================
    batch_parser = subparsers.add_parser('batch', help='Extract patches from multiple image-label pairs')
    batch_parser.add_argument('--data_dir', required=True, help='Directory containing image-label pairs')
    batch_parser.add_argument('--grid', required=True, help='Path to grid shapefile (shared for all images)')
    batch_parser.add_argument('--output', required=True, help='Base output directory')
    batch_parser.add_argument('--patch_size', type=int, default=224, help='Size of patches (default: 224)')
    batch_parser.add_argument('--image_channels', type=int, default=4, help='Number of image channels (default: 4)')
    batch_parser.add_argument('--label_channels', type=int, default=1, help='Number of label channels (default: 1)')
    batch_parser.add_argument('--train_ratio', type=float, default=0.75, help='Training ratio (default: 0.75)')
    batch_parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio (default: 0.2)')
    batch_parser.add_argument('--test_ratio', type=float, default=0.05, help='Test ratio (default: 0.05)')
    batch_parser.add_argument('--id_column', default='AUTO', help='Patch ID column (default: AUTO)')
    batch_parser.add_argument('--random_seed', type=int, help='Random seed for reproducible splits')
    batch_parser.add_argument('--interpolation', default='bilinear', 
                               choices=['nearest', 'bilinear', 'bicubic', 'lanczos'],
                               help='Interpolation method (default: bilinear)')
    batch_parser.add_argument('--no_compression', action='store_true', help='Disable LZW compression')
    batch_parser.add_argument('--recursive', action='store_true', help='Search data_dir recursively')
    batch_parser.add_argument('--image_pattern', default=r'^[Ii]mage[_-]?(\d+)\.tif{1,2}$',
                               help='Regex pattern for image files (default: Image_N.tif)')
    batch_parser.add_argument('--label_pattern', default=r'^[Ll]abel[_-]?(\d+)\.tif{1,2}$',
                               help='Regex pattern for label files (default: Label_N.tif)')
    batch_parser.add_argument('--use_per_image_grid', action='store_true',
                               help='Use Grid_X.shp for each Image_X.tif')
    batch_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # ========================
    # VISUALIZE
    # ========================
    viz_parser = subparsers.add_parser('visualize', help='Visualize extracted patches')
    viz_parser.add_argument('--output', required=True, help='Output directory from extraction')
    viz_parser.add_argument('--split', default='train', choices=['train', 'validation', 'test'],
                            help='Dataset split to visualize (default: train)')
    viz_parser.add_argument('--sample_index', type=int, default=0, help='Sample index to visualize')
    viz_parser.add_argument('--image_channels', type=int, default=4, help='Number of image channels')
    
    # ========================
    # INFO
    # ========================
    info_parser = subparsers.add_parser('info', help='Display dataset information')
    info_parser.add_argument('--output', required=True, help='Output directory from extraction')
    
    # Legacy 'extract' command (maps to 'single')
    extract_parser = subparsers.add_parser('extract', help='[LEGACY] Same as "single" mode')
    extract_parser.add_argument('--image', required=True, help='Path to input satellite image')
    extract_parser.add_argument('--label', required=True, help='Path to label image')
    extract_parser.add_argument('--grid', required=True, help='Path to grid shapefile')
    extract_parser.add_argument('--output', required=True, help='Base output directory')
    extract_parser.add_argument('--patch_size', type=int, default=224)
    extract_parser.add_argument('--image_channels', type=int, default=4)
    extract_parser.add_argument('--label_channels', type=int, default=1)
    extract_parser.add_argument('--train_ratio', type=float, default=0.75)
    extract_parser.add_argument('--val_ratio', type=float, default=0.2)
    extract_parser.add_argument('--test_ratio', type=float, default=0.05)
    extract_parser.add_argument('--id_column', default='AUTO')
    extract_parser.add_argument('--random_seed', type=int)
    extract_parser.add_argument('--interpolation', default='bilinear')
    extract_parser.add_argument('--no_compression', action='store_true')
    extract_parser.add_argument('--save_metadata', action='store_true')
    extract_parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # ========================
    # COMMAND HANDLING
    # ========================
    
    if args.command in ['single', 'extract']:
        # Validate ratios
        total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            print(f"❌ Error: Ratios must sum to 1.0 (got {total_ratio:.3f})")
            return 1
        
        print("🚀 Starting SINGLE mode patch extraction...")
        print(f"   Image: {args.image}")
        print(f"   Label: {args.label}")
        print(f"   Grid: {args.grid}")
        print(f"   Output: {args.output}")
        
        try:
            results = extract_patches_single(
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
            print(f"\n✅ Extraction completed!")
            print(f"\n📊 Statistics:")
            print(f"   Total patches: {stats['total_patches']}")
            print(f"   Train: {stats['train_patches']}")
            print(f"   Validation: {stats['val_patches']}")
            print(f"   Test: {stats['test_patches']}")
            print(f"   Resized: {stats['resized_patches']}")
            print(f"   Failed: {stats['failed_patches']}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1
    
    elif args.command == 'batch':
        # Validate ratios
        total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            print(f"❌ Error: Ratios must sum to 1.0 (got {total_ratio:.3f})")
            return 1
        
        print("🚀 Starting BATCH mode patch extraction...")
        print(f"   Data directory: {args.data_dir}")
        print(f"   Grid: {args.grid}")
        print(f"   Output: {args.output}")
        
        try:
            results = extract_patches_batch(
                data_dir=args.data_dir,
                grid_path=args.grid,
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
                save_metadata=True,
                recursive=args.recursive,
                image_pattern=args.image_pattern,
                label_pattern=args.label_pattern,
                use_per_image_grid=args.use_per_image_grid,
                verbose=args.verbose
            )
            
            stats = results['statistics']
            print(f"\n{'='*60}")
            print(f"✅ BATCH extraction completed!")
            print(f"\n📊 Aggregated Statistics:")
            print(f"   Pairs processed: {stats.get('pairs_processed', 0)}")
            print(f"   Total patches: {stats.get('total_patches', 0)}")
            print(f"   Train: {stats.get('train_patches', 0)}")
            print(f"   Validation: {stats.get('val_patches', 0)}")
            print(f"   Test: {stats.get('test_patches', 0)}")
            print(f"   Resized: {stats.get('resized_patches', 0)}")
            print(f"   Failed: {stats.get('failed_patches', 0)}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    elif args.command == 'visualize':
        try:
            visualize_sample(
                output_dir=args.output,
                split=args.split,
                sample_index=args.sample_index,
                image_channels=args.image_channels
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    elif args.command == 'info':
        list_dataset_info(args.output)
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
