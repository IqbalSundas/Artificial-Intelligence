"""
GRAD-CAM MODULE (Native PyTorch Version)
No external pytorch-grad-cam package needed!
Generates heatmap overlays showing which regions
the model focused on for its decision.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from config.config import IMG_SIZE, HEATMAP_DIR, DEVICE


# ──────────────────────────────────────────────
# NATIVE GRAD-CAM IMPLEMENTATION
# ──────────────────────────────────────────────
class _NativeGradCAM:
    """Custom Grad-CAM implementation using pure PyTorch."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass to get gradients
        self.model.zero_grad()
        output[0][target_class].backward(retain_graph=True)
        
        # Get captured data
        activations = self.activations.cpu().detach().numpy()[0]  # [C, H, W]
        gradients = self.gradients.cpu().detach().numpy()[0]      # [C, H, W]
        
        # Global average pooling of gradients to get weights
        weights = np.mean(gradients, axis=(1, 2))  # [C]
        
        # Weighted combination of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        return cam


def generate_gradcam(model, input_tensor, target_class=None):
    """
    Generates Grad-CAM heatmap for an input image.
    
    Args:
        model: Trained DeepFakeDetectorModel
        input_tensor: Preprocessed image tensor [1, 3, H, W]
        target_class: Which class to generate CAM for (1=Fake by default)
    
    Returns:
        numpy array: Grad-CAM heatmap (H, W) normalized to [0, 1]
    """
    # Target the final convolutional layer of EfficientNet-B0
    target_layer = model.backbone.conv_head
    
    cam_extractor = _NativeGradCAM(model, target_layer)
    heatmap = cam_extractor.generate(input_tensor.to(DEVICE), target_class)
    
    return heatmap


def overlay_gradcam_on_image(original_image, heatmap, alpha=0.5):
    """
    Overlays Grad-CAM heatmap on the original image.
    """
    original_image = original_image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(original_image)
    
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), 
        cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    blended = cv2.addWeighted(img_array, 1 - alpha, heatmap_color, alpha, 0)
    
    return Image.fromarray(blended)


def save_gradcam_result(original_image, heatmap, save_path=None, 
                         prediction=None, confidence=None):
    """
    Creates a side-by-side visualization:
    Original | Heatmap | Overlay
    """
    if save_path is None:
        save_path = HEATMAP_DIR / "gradcam_result.png"
    
    overlay = overlay_gradcam_on_image(original_image, heatmap)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image.resize((IMG_SIZE, IMG_SIZE)))
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    title = "Overlay"
    if prediction and confidence:
        title = f"Overlay\n{prediction} ({confidence:.1%})"
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grad-CAM saved to {save_path}")
    return overlay


def generate_gradcam_for_video(model, face_crops, output_dir=None):
    """
    Generates Grad-CAM heatmaps for multiple face crops from a video.
    """
    from src.preprocessing import preprocess_image
    
    output_dir = Path(output_dir) if output_dir else HEATMAP_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    model.eval()
    for i, face in enumerate(face_crops[:5]):  # Top 5 frames
        tensor = preprocess_image(face).to(DEVICE)
        
        with torch.no_grad():
            probs = model.predict_proba(tensor)
            pred = torch.argmax(probs, 1).item()
            conf = probs[0, pred].item()
        
        # Generate Grad-CAM for fake class
        heatmap = generate_gradcam(model, tensor, target_class=1)
        overlay = overlay_gradcam_on_image(face, heatmap)
        
        save_path = output_dir / f"frame_{i:04d}_gradcam.png"
        save_gradcam_result(face, heatmap, save_path, prediction="Fake" if pred == 1 else "Real", confidence=conf)
        
        results.append({
            'frame_index': i,
            'prediction': pred,
            'confidence': conf,
            'heatmap': heatmap,
            'overlay': overlay,
        })
    
    return results