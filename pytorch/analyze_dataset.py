#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze class distribution in tomato dataset
"""

import os
import sys
import glob
import numpy as np
import torch
from collections import Counter

def analyze_split(root, split):
    """Analyze class distribution in a dataset split."""
    split_root = os.path.join(root, split)
    pattern = os.path.join(split_root, '**', '*_inst_nostuff.pth')
    files = sorted(glob.glob(pattern, recursive=True))
    
    if len(files) == 0:
        print(f"No files found in {split} split")
        return
    
    print(f"\n{'='*60}")
    print(f"Analyzing {split.upper()} split: {len(files)} files")
    print(f"{'='*60}")
    
    all_labels = []
    class_counts = Counter()
    
    for i, file_path in enumerate(files):
        try:
            data = torch.load(file_path, map_location='cpu', weights_only=False)
        except TypeError:
            data = torch.load(file_path, map_location='cpu')
        
        # Extract semantic labels (third element in the tuple)
        semantic = data[2]
        if isinstance(semantic, torch.Tensor):
            labels = semantic.cpu().numpy().reshape(-1)
        else:
            labels = np.asarray(semantic).reshape(-1)
        
        all_labels.extend(labels.tolist())
        
        # Count per file
        unique, counts = np.unique(labels, return_counts=True)
        for cls, cnt in zip(unique, counts):
            class_counts[int(cls)] += cnt
    
    # Overall statistics
    all_labels = np.array(all_labels)
    total_points = len(all_labels)
    
    print(f"\nTotal points: {total_points:,}")
    print(f"\nClass distribution:")
    print(f"{'Class':<10} {'Count':<15} {'Percentage':<12} {'Label'}")
    print("-" * 60)
    
    class_names = {0: 'earth', 1: 'stem', 2: 'leaf'}
    
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        percentage = (count / total_points) * 100
        name = class_names.get(cls, f'unknown_{cls}')
        print(f"{cls:<10} {count:<15,} {percentage:<12.2f}% {name}")
    
    # Calculate class weights for balanced loss
    print(f"\n{'='*60}")
    print("Recommended class weights (inverse frequency):")
    print("-" * 60)
    
    max_count = max(class_counts.values())
    weights = {}
    for cls in sorted(class_counts.keys()):
        weight = max_count / class_counts[cls]
        weights[cls] = weight
        name = class_names.get(cls, f'unknown_{cls}')
        print(f"Class {cls} ({name}): {weight:.4f}")
    
    # Normalized weights
    print(f"\nNormalized weights (sum to num_classes):")
    num_classes = len(weights)
    weight_sum = sum(weights.values())
    normalized_weights = [weights[i] * num_classes / weight_sum for i in sorted(weights.keys())]
    print(f"weights = {normalized_weights}")
    
    return class_counts, weights, normalized_weights


if __name__ == '__main__':
    root = 'dataset/tomato'
    
    if not os.path.exists(root):
        print(f"Dataset root not found: {root}")
        sys.exit(1)
    
    # Analyze each split
    for split in ['train', 'val']:
        try:
            analyze_split(root, split)
        except Exception as e:
            print(f"Error analyzing {split}: {e}")
            import traceback
            traceback.print_exc()
