"""
EVALUATION SCRIPT
Evaluates the trained model with detailed metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, accuracy_score
)
from pathlib import Path

from config.config import MODEL_DIR, DEVICE
from src.dataset import create_dataloaders
from src.model import load_model


def evaluate_model():
    """Complete model evaluation pipeline."""
    
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION")
    print("="*60)
    
    # Load best model
    model_path = MODEL_DIR / "best_model.pth"
    if not model_path.exists():
        print("❌ No trained model found. Run training first!")
        return
    
    model = load_model(model_path)
    
    # Get test dataloader
    _, _, test_loader, class_names = create_dataloaders()
    
    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Fake probability
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # ── Classification Report ──
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # ── Confusion Matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'confusion_matrix.png', dpi=150)
    plt.show()
    
    # ── ROC Curve ──
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - DeepFake Detection')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'roc_curve.png', dpi=150)
    plt.show()
    
    # ── Overall accuracy ──
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n🎯 Test Accuracy: {acc*100:.2f}%")
    print(f"📈 AUC-ROC: {roc_auc:.3f}")
    
    return {'accuracy': acc, 'auc': roc_auc}


if __name__ == "__main__":
    evaluate_model()