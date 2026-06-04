"""
PREPROCESSING MODULE
- Extract frames from videos
- Detect faces using MTCNN
- Crop and normalize face regions
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

from config.config import (
    IMG_SIZE, FACE_DETECTION_CONF, 
    FRAMES_PER_VIDEO, MAX_FACES_PER_FRAME, DEVICE
)


# ──────────────────────────────────────────────
# MTCNN Face Detector (initialized once)
# ──────────────────────────────────────────────
mtcnn = MTCNN(
    image_size=IMG_SIZE,
    margin=40,
    min_face_size=30,
    thresholds=[0.6, 0.7, FACE_DETECTION_CONF],
    device=DEVICE,
    keep_all=False,  # Only detect the most prominent face
)


def extract_frames(video_path, num_frames=None, output_dir=None):
    """
    Extracts evenly-spaced frames from a video file.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default from config)
        output_dir: If provided, save frames as images here
    
    Returns:
        List of PIL Images
    """
    num_frames = num_frames or FRAMES_PER_VIDEO
    video_path = str(video_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    # Calculate frame indices (evenly spaced)
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            # Convert BGR → RGB → PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
            
            # Save if output_dir specified
            if output_dir:
                out_path = Path(output_dir) / f"frame_{idx:06d}.jpg"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pil_img.save(str(out_path))
    
    cap.release()
    print(f"  Extracted {len(frames)} frames from {Path(video_path).name}")
    print(f"  Video: {duration:.1f}s, {fps:.1f} FPS, {total_frames} total frames")
    
    return frames


def detect_and_crop_face(image, return_bbox=False):
    """
    Detects the largest face in an image and crops it.
    
    Args:
        image: PIL Image
        return_bbox: If True, also return bounding box
    
    Returns:
        Cropped face PIL Image (resized to IMG_SIZE x IMG_SIZE)
        Optional: bounding box [x1, y1, x2, y2]
    """
    # MTCNN detection
    box, prob = mtcnn.detect(image)
    
    if box is None or len(box) == 0 or prob[0] < 0.9:
        # No face detected — fallback: return resized full image
        face_crop = image.resize((IMG_SIZE, IMG_SIZE))
        if return_bbox:
            return face_crop, None
        return face_crop
    
    # Take the first (most confident) face
    bbox = box[0]
    x1, y1, x2, y2 = [int(c) for c in bbox]
    
    # Add margin
    w, h = image.size
    margin = 20
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    
    # Crop and resize
    face_crop = image.crop((x1, y1, x2, y2))
    face_crop = face_crop.resize((IMG_SIZE, IMG_SIZE))
    
    if return_bbox:
        return face_crop, [x1, y1, x2, y2]
    return face_crop


def preprocess_image(image):
    """
    Normalizes a PIL Image for model input.
    Returns a PyTorch tensor of shape [1, 3, H, W].
    """
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


def extract_audio_from_video(video_path, output_path=None):
    """
    Extracts audio from a video file and saves as WAV.
    Uses moviepy to separate audio track.
    
    Args:
        video_path: Path to video file
        output_path: Path for output WAV file
    
    Returns:
        Path to extracted audio file, or None if no audio
    """
    from moviepy.editor import VideoFileClip
    
    video_path = str(video_path)
    
    if output_path is None:
        output_path = str(Path(video_path).with_suffix('.wav'))
    
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            print("  ⚠️  No audio track found in video")
            video.close()
            return None
        
        video.audio.write_audiofile(output_path, verbose=False, logger=None)
        video.close()
        print(f"  ✅ Audio extracted: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"  ❌ Audio extraction failed: {e}")
        return None


def process_video_pipeline(video_path, output_dir=None):
    """
    COMPLETE VIDEO PROCESSING PIPELINE:
    1. Extract frames
    2. Detect faces in each frame
    3. Extract audio
    
    Returns:
        dict with keys: 'frames', 'face_crops', 'bboxes', 'audio_path'
    """
    print(f"\n🎬 Processing video: {Path(video_path).name}")
    
    # 1. Extract frames
    frames = extract_frames(video_path, output_dir=output_dir)
    if not frames:
        return None
    
    # 2. Detect faces in each frame
    face_crops = []
    bboxes = []
    for frame in frames:
        face, bbox = detect_and_crop_face(frame, return_bbox=True)
        face_crops.append(face)
        bboxes.append(bbox)
    
    print(f"  Detected faces in {sum(1 for b in bboxes if b is not None)}/{len(frames)} frames")
    
    # 3. Extract audio
    audio_path = extract_audio_from_video(video_path)
    
    return {
        'frames': frames,
        'face_crops': face_crops,
        'bboxes': bboxes,
        'audio_path': audio_path,
        'video_path': str(video_path),
    }