#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-time training monitor for DGCNN
"""

import os
import sys
import time
import re

def tail_file(filepath, n=50):
    """Get last n lines of a file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            return lines[-n:]
    except FileNotFoundError:
        return []

def parse_training_log(lines):
    """Parse training log and extract key metrics."""
    train_pattern = r'Train\s+(\d+),\s+loss:\s+([\d.]+),\s+acc:\s+([\d.]+),\s+mIoU:\s+([\d.]+)'
    val_pattern = r'Val\s+(\d+),\s+loss:\s+([\d.]+),\s+acc:\s+([\d.]+),\s+mIoU:\s+([\d.]+),\s+IoU per class:\s+\[(.*?)\]'
    
    train_history = []
    val_history = []
    
    for line in lines:
        train_match = re.search(train_pattern, line)
        if train_match:
            epoch, loss, acc, miou = train_match.groups()
            train_history.append({
                'epoch': int(epoch),
                'loss': float(loss),
                'acc': float(acc),
                'miou': float(miou)
            })
        
        val_match = re.search(val_pattern, line)
        if val_match:
            epoch, loss, acc, miou, iou_str = val_match.groups()
            # Parse IoU per class
            iou_values = [float(x.strip()) if x.strip() and x.strip() != 'nan' else 0.0 
                         for x in iou_str.split()]
            val_history.append({
                'epoch': int(epoch),
                'loss': float(loss),
                'acc': float(acc),
                'miou': float(miou),
                'iou_per_class': iou_values
            })
    
    return train_history, val_history

def print_summary(train_history, val_history):
    """Print a nice summary of training progress."""
    if not val_history:
        print("No validation data yet...")
        return
    
    latest_val = val_history[-1]
    epoch = latest_val['epoch']
    
    print("\n" + "="*70)
    print(f"  Training Summary - Epoch {epoch}")
    print("="*70)
    
    # Current performance
    print(f"\n📊 Current Performance (Epoch {epoch}):")
    print(f"  Validation Loss: {latest_val['loss']:.6f}")
    print(f"  Validation Acc:  {latest_val['acc']:.2%}")
    print(f"  Validation mIoU: {latest_val['miou']:.2%}")
    
    print(f"\n  IoU per Class:")
    class_names = ['Class 0 (earth)', 'Class 1 (stem)', 'Class 2 (leaf)']
    iou = latest_val['iou_per_class']
    for i, (name, val) in enumerate(zip(class_names, iou)):
        bar_length = int(val * 50)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"    {name:20s}: {bar} {val:.2%}")
    
    # Best performance
    best_miou_idx = max(range(len(val_history)), key=lambda i: val_history[i]['miou'])
    best_val = val_history[best_miou_idx]
    
    if best_val['epoch'] == epoch:
        print(f"\n🎉 NEW BEST mIoU!")
    else:
        print(f"\n🏆 Best mIoU so far: {best_val['miou']:.2%} (Epoch {best_val['epoch']})")
    
    # Stem class progress
    if len(val_history) >= 5:
        recent_stem_iou = [v['iou_per_class'][1] for v in val_history[-5:]]
        print(f"\n📈 Class 1 (Stem) Progress (last 5 epochs):")
        for i, iou_val in enumerate(recent_stem_iou):
            epoch_num = val_history[-(5-i)]['epoch']
            bar_length = int(iou_val * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            print(f"    Epoch {epoch_num:3d}: {bar} {iou_val:.2%}")
    
    # Warnings
    print(f"\n⚠️  Analysis:")
    if iou[1] < 0.3:
        print(f"    • Class 1 (stem) IoU is low ({iou[1]:.2%})")
        print(f"      → This is likely due to class imbalance")
        print(f"      → Consider using class weights in the loss function")
    if latest_val['miou'] < 0.6:
        print(f"    • Overall mIoU is below 60%")
        print(f"      → Model may need more training epochs")
    
    print("\n" + "="*70)

def monitor_training(log_path, interval=10):
    """Monitor training in real-time."""
    print(f"Monitoring: {log_path}")
    print(f"Update interval: {interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_size = 0
    
    try:
        while True:
            if os.path.exists(log_path):
                current_size = os.path.getsize(log_path)
                if current_size != last_size:
                    lines = tail_file(log_path, n=100)
                    train_history, val_history = parse_training_log(lines)
                    
                    os.system('clear' if os.name != 'nt' else 'cls')
                    print_summary(train_history, val_history)
                    
                    last_size = current_size
            else:
                print(f"Waiting for log file: {log_path}")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == '__main__':
    log_path = 'checkpoints/tomato_seg_no_ignore/run.log'
    
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    
    monitor_training(log_path, interval=5)
