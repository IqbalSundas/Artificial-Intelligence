"""
TRAINING SCRIPT
Trains the EfficientNet-B4 model on the deepfake dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import time
from tqdm import tqdm

from config.config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, 
    EARLY_STOP_PATIENCE, DEVICE, MODEL_DIR, BATCH_SIZE
)
from src.dataset import create_dataloaders
from src.model import DeepFakeDetectorModel, save_model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains model for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="  Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validates model. Returns average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="  Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model():
    """
    COMPLETE TRAINING PIPELINE
    1. Create dataloaders
    2. Initialize model, loss, optimizer
    3. Train for NUM_EPOCHS
    4. Save best model
    """
    print("\n" + "="*60)
    print("🏋️  STARTING TRAINING")
    print("="*60)
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print()
    
    # 1. Create dataloaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders()
    
    # 2. Initialize model
    model = DeepFakeDetectorModel().to(DEVICE)
    
    # Handle class imbalance with weighted loss
    # Count real vs fake in training set
    num_real = sum(1 for _, label in train_loader.dataset if label == 0)
    num_fake = len(train_loader.dataset) - num_real
    class_weights = torch.tensor([1.0/num_real, 1.0/num_fake])
    class_weights = class_weights / class_weights.sum() * 2
    class_weights = class_weights.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # 3. Training loop
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n📌 Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        elapsed = time.time() - start_time
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"  Time: {elapsed:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, MODEL_DIR / "best_model.pth")
            print(f"  🏆 New best model! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
        
        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
    
    # 4. Final test evaluation
    print("\n" + "="*60)
    print("📊 FINAL TEST EVALUATION")
    print("="*60)
    
    # Load best model
    from src.model import load_model
    best_model = load_model(MODEL_DIR / "best_model.pth")
    
    test_loss, test_acc = validate(best_model, test_loader, criterion, DEVICE)
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    
    # Save training history
    import json
    history_path = MODEL_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_history(history)
    
    return best_model, history


def plot_training_history(history):
    """Plots training and validation loss/accuracy curves."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_curves.png", dpi=150)
    plt.show()
    print(f"✅ Training curves saved to {MODEL_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    train_model()