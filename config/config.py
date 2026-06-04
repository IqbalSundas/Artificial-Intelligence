"""
CENTRAL CONFIGURATION FILE
All paths, hyperparameters, and settings live here.
Change these values to customize the project.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# PROJECT ROOT (auto-detected)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
TEST_VIDEO_DIR  = DATA_DIR / "test_videos"
MODEL_DIR       = PROJECT_ROOT / "models"
OUTPUT_DIR      = PROJECT_ROOT / "outputs"
HEATMAP_DIR     = OUTPUT_DIR / "heatmaps"
REPORT_DIR      = OUTPUT_DIR / "reports"

# Create directories if they don't exist
for d in [RAW_DIR, PROCESSED_DIR, TEST_VIDEO_DIR, MODEL_DIR, 
          HEATMAP_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# KAGGLE DATASET
# ──────────────────────────────────────────────
KAGGLE_DATASET_SLUG = "manjilkarki/deepfake-and-real-images"
# Alternative datasets (change slug above if needed):
#   "xhlulu/140k-real-and-fake-faces"
#   "dagnelies/deepfake-faces"
#   "ciplab/real-and-fake-face-detection"

# After download, the folder structure is usually:
#   data/raw/Dataset/Real/   and   data/raw/Dataset/Fake/
# Adjust these if your dataset has different folder names:
REAL_FOLDER_NAME = "Real"
FAKE_FOLDER_NAME = "Fake"

# ──────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────
IMG_SIZE            = 224          # EfficientNet-B4 input size
FACE_DETECTION_CONF = 0.95         # MTCNN confidence threshold
FRAMES_PER_VIDEO    = 15           # How many frames to extract per video
MAX_FACES_PER_FRAME = 1           # Only take the largest face

# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
MODEL_NAME          = "efficientnet_b0"
NUM_CLASSES         = 2            # Real vs Fake
PRETRAINED          = True
DROPOUT_RATE        = 0.3

# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
BATCH_SIZE          = 8
NUM_EPOCHS          = 3
LEARNING_RATE       = 1e-4
WEIGHT_DECAY        = 1e-5
TRAIN_SPLIT         = 0.8
VAL_SPLIT           = 0.1
TEST_SPLIT          = 0.1
EARLY_STOP_PATIENCE = 3

# ──────────────────────────────────────────────
# ECF MODULE (Emotion-Consistency Forensics)
# ──────────────────────────────────────────────
WHISPER_MODEL_SIZE  = "small"      # tiny, base, small, medium, large
ECF_WEIGHT          = 0.4          # Weight for ECF score in fusion
VISUAL_WEIGHT       = 0.6          # Weight for visual CNN score

# Emotion-Sentiment mapping for cross-checking
EMOTION_SENTIMENT_MAP = {
    "happy":     "positive",
    "surprise":  "positive",
    "neutral":   "neutral",
    "sad":       "negative",
    "angry":     "negative",
    "fear":      "negative",
    "disgust":   "negative",
}

# ──────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# LABELS
# ──────────────────────────────────────────────
LABEL_MAP = {0: "Real", 1: "Fake"}