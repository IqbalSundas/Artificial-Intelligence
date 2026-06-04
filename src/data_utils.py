"""
DATA DOWNLOAD & LOADING UTILITIES
Downloads dataset from Kaggle using the Kaggle API.
"""

import os
import shutil
import zipfile
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from config.config import (
    RAW_DIR, KAGGLE_DATASET_SLUG, 
    REAL_FOLDER_NAME, FAKE_FOLDER_NAME
)


def setup_kaggle_auth(kaggle_json_path: str = None):
    """
    Sets up Kaggle API authentication.
    
    Steps:
      1. Download kaggle.json from https://www.kaggle.com/settings
      2. Place it in the project root OR provide its path
      3. This function copies it to ~/.kaggle/
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    target = kaggle_dir / "kaggle.json"
    
    if target.exists():
        print("✅ Kaggle auth already configured.")
        return
    
    if kaggle_json_path is None:
        # Try project root
        project_json = Path.cwd() / "kaggle.json"
        if project_json.exists():
            kaggle_json_path = str(project_json)
        else:
            print("❌ kaggle.json not found!")
            print("   1. Go to https://www.kaggle.com/settings")
            print("   2. Click 'Create New Token'")
            print("   3. Place kaggle.json in this project folder")
            raise FileNotFoundError("kaggle.json missing")
    
    shutil.copy(kaggle_json_path, str(target))
    os.chmod(str(target), 0o600)
    print(f"✅ Kaggle auth configured: {target}")


def download_dataset(slug: str = None, dest: Path = None):
    """
    Downloads and unzips a Kaggle dataset.
    
    Args:
        slug: Kaggle dataset slug (e.g. 'manjilkarki/deepfake-and-real-images')
        dest: Destination directory
    """
    slug = slug or KAGGLE_DATASET_SLUG
    dest = dest or RAW_DIR
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    marker = dest / ".download_complete"
    if marker.exists():
        print(f"✅ Dataset already downloaded at {dest}")
        return dest
    
    print(f"⬇️  Downloading dataset: {slug}")
    print(f"   Destination: {dest}")
    print("   This may take several minutes depending on dataset size...")
    
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(dest), unzip=True)
    
    # Mark as downloaded
    marker.touch()
    print(f"✅ Dataset downloaded and extracted to {dest}")
    
    return dest


def find_image_folders(base_dir: Path):
    """
    Automatically finds REAL and FAKE image folders regardless of
    exact folder names in the downloaded dataset.
    """
    base_dir = Path(base_dir)
    
    real_dir = None
    fake_dir = None
    
    # Search recursively for folders with common names
    real_names = ["real", "Real", "REAL", "training_real", "0_real"]
    fake_names = ["fake", "Fake", "FAKE", "training_fake", "1_fake"]
    
    for root, dirs, files in os.walk(base_dir):
        folder_name = Path(root).name
        if folder_name in real_names and real_dir is None:
            real_dir = Path(root)
        if folder_name in fake_names and fake_dir is None:
            fake_dir = Path(root)
    
    return real_dir, fake_dir


def get_image_paths(folder: Path, extensions=None):
    """
    Returns list of image file paths from a folder.
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    folder = Path(folder)
    if not folder.exists():
        return []
    
    paths = [
        p for p in folder.rglob("*") 
        if p.suffix.lower() in extensions
    ]
    return sorted(paths)


def print_dataset_stats(real_dir: Path, fake_dir: Path):
    """Prints simple stats about the dataset."""
    real_images = get_image_paths(real_dir) if real_dir else []
    fake_images = get_image_paths(fake_dir) if fake_dir else []
    
    print("\n" + "="*50)
    print("📊 DATASET STATISTICS")
    print("="*50)
    print(f"  Real images: {len(real_images)}")
    print(f"  Fake images: {len(fake_images)}")
    print(f"  Total:       {len(real_images) + len(fake_images)}")
    print(f"  Ratio (R:F): {len(real_images)/max(len(fake_images),1):.2f}")
    print("="*50 + "\n")
    
    return len(real_images), len(fake_images)