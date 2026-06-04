"""
UTILITY FUNCTIONS
Shared helpers used across the project.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
import time


def set_seed(seed=42):
    """Sets random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    """Counts trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
    return total


def save_report(report_dict, filename, output_dir=None):
    """Saves a report dictionary as JSON."""
    from config.config import REPORT_DIR
    output_dir = Path(output_dir) if output_dir else REPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert non-serializable types
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Image.Image):
            return "<PIL Image>"
        return obj
    
    clean_dict = json.loads(
        json.dumps(report_dict, default=convert)
    )
    
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(clean_dict, f, indent=2)
    
    print(f"✅ Report saved to {filepath}")


class Timer:
    """Simple timer context manager."""
    def __init__(self, name="Process"):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        print(f"⏱️  {self.name} started...")
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        if elapsed > 60:
            print(f"✅ {self.name} completed in {elapsed/60:.1f} minutes")
        else:
            print(f"✅ {self.name} completed in {elapsed:.1f} seconds")