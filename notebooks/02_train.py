"""
TRAINING SCRIPT
Simply calls the training pipeline from src/train.py
"""

from src.train import train_model

if __name__ == "__main__":
    model, history = train_model()