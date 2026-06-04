"""
GRAD-CAM MODULE (Native PyTorch Version - No external packages needed)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from config.config import IMG_SIZE, HEATMAP_DIR, DEVICE

class _NativeGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0][target_class].backward(retain_graph=True)
        activations = self.activations.cpu().detach().numpy()[0]
        gradients = self.gradients.cpu().detach().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        return cam

def generate_gradcam(model, input_tensor, target_class=None):
    target_layer = model.backbone.conv_head
    cam_extractor = _NativeGradCAM(model, target_layer)
    heatmap = cam_extractor.generate(input_tensor.to(DEVICE), target_class)
    return heatmap

def overlay_gradcam_on_image(original_image, heatmap, alpha=0.5):
    original_image = original_image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(original_image)
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_array, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(blended)

def save_gradcam_result(original_image, heatmap, save_path=None, prediction=None, confidence=None):
    if save_path is None:
        save_path = HEATMAP_DIR / "gradcam_result.png"
    overlay = overlay_gradcam_on_image(original_image, heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image.resize((IMG_SIZE, IMG_SIZE))); axes[0].set_title("Original"); axes[0].axis('off')
    axes[1].imshow(heatmap, cmap='jet'); axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis('off')
    axes[2].imshow(overlay)
    title = "Overlay"
    if prediction and confidence: title = f"Overlay\n{prediction} ({confidence:.1%})"
    axes[2].set_title(title); axes[2].axis('off')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return overlay
