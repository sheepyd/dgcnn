#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test data loading speed
"""

import time
import sys
sys.path.insert(0, '.')

from data import TomatoDataset
from torch.utils.data import DataLoader

print("Testing data loading...")
print("Creating dataset...")
train_dataset = TomatoDataset(root='dataset/tomato', split='train', num_points=2048, augment=True)
print(f"Dataset size: {len(train_dataset)} files")

print("\nTesting single sample load...")
start = time.time()
sample = train_dataset[0]
elapsed = time.time() - start
print(f"Single sample loaded in {elapsed:.2f}s")
print(f"Points shape: {sample[0].shape}, Labels shape: {sample[1].shape}")

print("\nTesting DataLoader (num_workers=0)...")
loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)
start = time.time()
for i, (data, label) in enumerate(loader):
    elapsed = time.time() - start
    print(f"Batch {i}: {data.shape}, {label.shape}, time: {elapsed:.2f}s")
    if i >= 2:  # Test first 3 batches
        break
    start = time.time()

print("\nData loading test completed!")
