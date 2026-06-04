"""
EFFICIENTNET-B4 MODEL
Pre-trained model fine-tuned for DeepFake Detection.
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path

from config.config import MODEL_NAME, NUM_CLASSES, PRETRAINED, DROPOUT_RATE, DEVICE


class DeepFakeDetectorModel(nn.Module):
    """
    EfficientNet-B4 based binary classifier (Real vs Fake).
    Uses pre-trained ImageNet weights and replaces the head.
    """
    
    def __init__(self, model_name=None, num_classes=None, pretrained=None):
        super().__init__()
        
        model_name = model_name or MODEL_NAME
        num_classes = num_classes or NUM_CLASSES
        pretrained = pretrained if pretrained is not None else PRETRAINED
        
        # Load pre-trained EfficientNet-B4 from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
        
        print(f"✅ Model created: {model_name}")
        print(f"   Feature dim: {self.feature_dim}")
        print(f"   Num classes: {num_classes}")
        print(f"   Device: {DEVICE}")
    
    def forward(self, x):
        """Forward pass. Returns raw logits."""
        features = self.backbone(x)        # [B, feature_dim]
        logits = self.classifier(features)  # [B, num_classes]
        return logits
    
    def predict_proba(self, x):
        """Returns softmax probabilities."""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs
    
    def predict(self, x):
        """Returns predicted class labels."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)
    
    def get_fake_probability(self, x):
        """
        Returns the probability of being FAKE.
        Useful for the fusion module.
        """
        probs = self.predict_proba(x)
        return probs[:, 1]  # Index 1 = Fake class


def save_model(model, path):
    """Saves model weights."""
    path = Path(path) if not isinstance(path, Path) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': MODEL_NAME,
        'num_classes': NUM_CLASSES,
    }, str(path))
    print(f"✅ Model saved to {path}")


def load_model(path, device=None):
    """Loads model weights."""
    device = device or DEVICE
    checkpoint = torch.load(str(path), map_location=device)
    
    model = DeepFakeDetectorModel(
        model_name=checkpoint.get('model_name', MODEL_NAME),
        num_classes=checkpoint.get('num_classes', NUM_CLASSES),
        pretrained=False,  # Don't download weights again
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from {path}")
    return model