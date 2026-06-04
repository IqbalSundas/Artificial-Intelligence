"""
PYTORCH DATASET CLASS
Loads real/fake face images for training and validation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

from config.config import (
    IMG_SIZE, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    REAL_FOLDER_NAME, FAKE_FOLDER_NAME, RAW_DIR
)
from src.data_utils import find_image_folders, get_image_paths


class DeepFakeDataset(Dataset):
    """
    Custom PyTorch Dataset for DeepFake Detection.
    
    Labels: 0 = Real, 1 = Fake
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (0=Real, 1=Fake)
            transform: Torchvision transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or self._default_transform()
    
    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def val_transform():
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return a black image if file is corrupt
            print(f"⚠️  Cannot load {img_path}: {e}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_dataloaders(batch_size=None, data_dir=None):
    """
    Creates train, validation, and test DataLoaders.
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    batch_size = batch_size or BATCH_SIZE
    data_dir = Path(data_dir) if data_dir else RAW_DIR
    
    # Find real and fake folders
    real_dir, fake_dir = find_image_folders(data_dir)
    
    if real_dir is None or fake_dir is None:
        raise ValueError(
            f"Could not find Real/Fake folders in {data_dir}\n"
            f"Found: real={real_dir}, fake={fake_dir}\n"
            f"Check your dataset structure in config.py"
        )
    
    # Get image paths
    real_paths = get_image_paths(real_dir)
    fake_paths = get_image_paths(fake_dir)
    
    # Create labels
    real_labels = [0] * len(real_paths)   # 0 = Real
    fake_labels = [1] * len(fake_paths)   # 1 = Fake
    
    all_paths = real_paths + fake_paths
    all_labels = real_labels + fake_labels
    
    print(f"\n📂 Dataset loaded: {len(real_paths)} Real + {len(fake_paths)} Fake = {len(all_paths)} Total")
    
    # Split: 80% train, 10% val, 10% test
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, 
        test_size=(1 - TRAIN_SPLIT), 
        random_state=42,
        stratify=all_labels
    )
    
    val_ratio = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio),
        random_state=42,
        stratify=temp_labels
    )
    
    print(f"  Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Create datasets
    train_dataset = DeepFakeDataset(
        train_paths, train_labels, 
        transform=DeepFakeDataset._default_transform()  # with augmentation
    )
    val_dataset = DeepFakeDataset(
        val_paths, val_labels, 
        transform=DeepFakeDataset.val_transform()
    )
    test_dataset = DeepFakeDataset(
        test_paths, test_labels, 
        transform=DeepFakeDataset.val_transform()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    class_names = ["Real", "Fake"]
    
    return train_loader, val_loader, test_loader, class_names