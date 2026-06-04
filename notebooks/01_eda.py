"""
EDA (EXPLORATORY DATA ANALYSIS)
Run this after downloading the dataset.
Analyzes the deepfake dataset with visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import Counter

from config.config import RAW_DIR, PROCESSED_DIR
from src.data_utils import (
    download_dataset, setup_kaggle_auth,
    find_image_folders, get_image_paths, print_dataset_stats
)


def run_eda():
    """Complete EDA pipeline."""
    
    print("\n" + "="*60)
    print("📊 EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # ── Step 1: Download data ──
    print("\n⬇️  Step 1: Downloading dataset from Kaggle...")
    setup_kaggle_auth()
    download_dataset()
    
    # ── Step 2: Find folders ──
    print("\n📂 Step 2: Locating image folders...")
    real_dir, fake_dir = find_image_folders(RAW_DIR)
    print(f"   Real folder: {real_dir}")
    print(f"   Fake folder: {fake_dir}")
    
    if real_dir is None or fake_dir is None:
        print("❌ Could not find Real/Fake folders. Check dataset structure.")
        return
    
    # ── Step 3: Basic stats ──
    print("\n📊 Step 3: Dataset statistics...")
    real_paths = get_image_paths(real_dir)
    fake_paths = get_image_paths(fake_dir)
    num_real, num_fake = print_dataset_stats(real_dir, fake_dir)
    
    # ── Step 4: Image dimensions analysis ──
    print("\n📏 Step 4: Analyzing image dimensions...")
    analyze_image_dimensions(real_paths[:500], fake_paths[:500])
    
    # ── Step 5: Pixel intensity distributions ──
    print("\n🎨 Step 5: Analyzing pixel distributions...")
    analyze_pixel_distributions(real_paths[:200], fake_paths[:200])
    
    # ── Step 6: Sample visualizations ──
    print("\n🖼️  Step 6: Displaying sample images...")
    show_sample_images(real_paths, fake_paths)
    
    # ── Step 7: Class balance ──
    print("\n⚖️  Step 7: Checking class balance...")
    plot_class_balance(num_real, num_fake)
    
    print("\n✅ EDA Complete!")


def analyze_image_dimensions(real_paths, fake_paths, max_samples=500):
    """Analyzes width, height, and aspect ratio distributions."""
    dims = {'real_w': [], 'real_h': [], 'fake_w': [], 'fake_h': []}
    
    for path in real_paths[:max_samples]:
        try:
            img = Image.open(path)
            dims['real_w'].append(img.width)
            dims['real_h'].append(img.height)
        except: pass
    
    for path in fake_paths[:max_samples]:
        try:
            img = Image.open(path)
            dims['fake_w'].append(img.width)
            dims['fake_h'].append(img.height)
        except: pass
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, (label, data) in enumerate([
        ('Real Width', dims['real_w']),
        ('Real Height', dims['real_h']),
        ('Fake Width', dims['fake_w']),
        ('Fake Height', dims['fake_h']),
    ]):
        ax = axes[i // 2][i % 2]
        ax.hist(data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_title(label)
        ax.set_xlabel('Pixels')
        ax.set_ylabel('Count')
        ax.axvline(np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.0f}')
        ax.legend()
    
    plt.suptitle('Image Dimension Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RAW_DIR / 'eda_dimensions.png', dpi=150)
    plt.show()


def analyze_pixel_distributions(real_paths, fake_paths, max_samples=200):
    """Compares RGB pixel intensity distributions between real and fake."""
    real_pixels = {'R': [], 'G': [], 'B': []}
    fake_pixels = {'R': [], 'G': [], 'B': []}
    
    for path in real_paths[:max_samples]:
        try:
            img = np.array(Image.open(path).resize((224, 224)))
            real_pixels['R'].extend(img[:,:,0].flatten())
            real_pixels['G'].extend(img[:,:,1].flatten())
            real_pixels['B'].extend(img[:,:,2].flatten())
        except: pass
    
    for path in fake_paths[:max_samples]:
        try:
            img = np.array(Image.open(path).resize((224, 224)))
            fake_pixels['R'].extend(img[:,:,0].flatten())
            fake_pixels['G'].extend(img[:,:,1].flatten())
            fake_pixels['B'].extend(img[:,:,2].flatten())
        except: pass
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['red', 'green', 'blue']
    
    for i, channel in enumerate(['R', 'G', 'B']):
        ax = axes[i]
        # Subsample for plotting speed
        r_sample = np.random.choice(real_pixels[channel], min(10000, len(real_pixels[channel])), replace=False)
        f_sample = np.random.choice(fake_pixels[channel], min(10000, len(fake_pixels[channel])), replace=False)
        
        ax.hist(r_sample, bins=50, alpha=0.5, label='Real', color='green', density=True)
        ax.hist(f_sample, bins=50, alpha=0.5, label='Fake', color='red', density=True)
        ax.set_title(f'{channel} Channel Distribution')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.suptitle('RGB Pixel Intensity: Real vs Fake', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RAW_DIR / 'eda_pixel_distributions.png', dpi=150)
    plt.show()


def show_sample_images(real_paths, fake_paths, num=5):
    """Displays sample real and fake images side by side."""
    fig, axes = plt.subplots(2, num, figsize=(20, 8))
    
    for i in range(num):
        # Real
        idx = np.random.randint(0, len(real_paths))
        img = Image.open(real_paths[idx])
        axes[0][i].imshow(img)
        axes[0][i].set_title(f"Real #{i+1}")
        axes[0][i].axis('off')
        
        # Fake
        idx = np.random.randint(0, len(fake_paths))
        img = Image.open(fake_paths[idx])
        axes[1][i].imshow(img)
        axes[1][i].set_title(f"Fake #{i+1}")
        axes[1][i].axis('off')
    
    plt.suptitle('Sample Images: Real (Top) vs Fake (Bottom)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RAW_DIR / 'eda_samples.png', dpi=150)
    plt.show()


def plot_class_balance(num_real, num_fake):
    """Plots class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ['Real', 'Fake']
    counts = [num_real, num_fake]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='black', width=0.5)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Number of Images')
    ax.set_title('Class Distribution: Real vs Fake', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RAW_DIR / 'eda_class_balance.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    run_eda()