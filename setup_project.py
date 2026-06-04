"""
MASTER SETUP SCRIPT
Run this FIRST to set up everything:
  1. Installs dependencies
  2. Downloads dataset from Kaggle
  3. Verifies everything is working
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Runs a shell command and prints status."""
    print(f"\n{'='*50}")
    print(f"🔧 {description}")
    print(f"{'='*50}")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"⚠️  Command returned code {result.returncode}")
    else:
        print(f"✅ {description} — Done!")
    
    return result.returncode


def main():
    print("""
    ╔══════════════════════════════════════════════╗
    ║   DeepFake Detector — Project Setup          ║
    ║   AI-Powered Multimedia Authenticity         ║
    ║   Analysis System                            ║
    ╚══════════════════════════════════════════════╝
    """)
    
    # Step 1: Create directories
    print("\n📁 Step 1: Creating project directories...")
    dirs = [
        "data/raw", "data/processed", "data/test_videos",
        "models", "outputs/heatmaps", "outputs/reports",
        "config", "src", "notebooks"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created!")
    
    # Step 2: Install packages
    print("\n📦 Step 2: Installing Python packages...")
    print("   (This may take 5-10 minutes)")
    
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "timm",
        "facenet-pytorch",
        "fer",
        "openai-whisper",
        "nltk",
        "vaderSentiment",
        "grad-cam",
        "streamlit",
        "kaggle",
        "opencv-python",
        "Pillow",
        "moviepy",
        "librosa",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "tqdm",
    ]
    
    for pkg in packages:
        run_command(f"pip install {pkg}", f"Installing {pkg.split('==')[0]}")
    
    # Step 3: Setup Kaggle authentication
    print("\n🔑 Step 3: Setting up Kaggle authentication...")
    kaggle_json = Path("kaggle.json")
    if kaggle_json.exists():
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        
        import shutil
        shutil.copy("kaggle.json", str(kaggle_dir / "kaggle.json"))
        os.chmod(str(kaggle_dir / "kaggle.json"), 0o600)
        print("✅ kaggle.json configured!")
    else:
        print("⚠️  kaggle.json not found in project root!")
        print("   Please download it from https://www.kaggle.com/settings")
        print("   Place it in the project folder and re-run this script.")
    
    # Step 4: Download dataset
    print("\n⬇️  Step 4: Downloading dataset from Kaggle...")
    try:
        from src.data_utils import setup_kaggle_auth, download_dataset
        setup_kaggle_auth()
        download_dataset()
    except Exception as e:
        print(f"⚠️  Dataset download failed: {e}")
        print("   You can download manually from Kaggle and extract to data/raw/")
    
    # Step 5: Verify installation
    print("\n✅ Step 5: Verifying installation...")
    verify_installation()
    
    print("""
    ╔══════════════════════════════════════════════╗
    ║   ✅ SETUP COMPLETE!                         ║
    ║                                              ║
    ║   Next steps:                                ║
    ║   1. Run EDA:    python notebooks/01_eda.py  ║
    ║   2. Train:      python notebooks/02_train.py║
    ║   3. Evaluate:   python notebooks/03_eval.py ║
    ║   4. Launch App: streamlit run app.py        ║
    ╚══════════════════════════════════════════════╝
    """)


def verify_installation():
    """Verifies that all required packages are installed."""
    required = {
        'torch': 'PyTorch',
        'timm': 'Timm (EfficientNet)',
        'facenet_pytorch': 'FaceNet-PyTorch (MTCNN)',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'sklearn': 'Scikit-Learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
    }
    
    optional = {
        'fer': 'FER (Facial Emotion)',
        'whisper': 'OpenAI Whisper',
        'vaderSentiment': 'VADER Sentiment',
        'pytorch_grad_cam': 'Grad-CAM',
        'streamlit': 'Streamlit',
        'kaggle': 'Kaggle API',
    }
    
    print("\n  Required packages:")
    for module, name in required.items():
        try:
            __import__(module)
            print(f"    ✅ {name}")
        except ImportError:
            print(f"    ❌ {name} — MISSING!")
    
    print("\n  Optional packages:")
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"    ✅ {name}")
        except ImportError:
            print(f"    ⚠️  {name} — not installed yet")


if __name__ == "__main__":
    main()