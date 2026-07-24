"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (2017).

This hooks into the last convolutional block of the DenseNet121 backbone
to produce a class-discriminative localization heatmap explaining which
regions of the X-ray most influenced the model's prediction.
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.ml.model import ChestXrayClassifier


class GradCAM:
    def __init__(self, model: ChestXrayClassifier, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.model.eval()
        # Default target: last conv layer of the DenseNet feature extractor
        self.target_layer = target_layer or self._get_last_conv_layer()
        self._activations = None
        self._gradients = None
        self._register_hooks()

    def _get_last_conv_layer(self):
        # DenseNet121's final norm layer just before the classifier head
        return self.model.features.norm5

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self._activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None) -> Tuple[np.ndarray, int, float]:
        """
        Runs a forward+backward pass and returns:
            (heatmap [H,W] in [0,1], predicted_class_idx, confidence)
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, target_class].item())

        score = logits[0, target_class]
        score.backward()

        activations = self._activations[0]      # (C, H, W)
        gradients = self._gradients[0]           # (C, H, W)

        weights = gradients.mean(dim=(1, 2))     # (C,) global-average-pooled gradients
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy(), target_class, confidence


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Resizes heatmap to the original image size and overlays it as a color map."""
    import matplotlib.cm as cm

    orig_resized = original_image.convert("RGB").resize((224, 224))
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize((224, 224), resample=Image.BILINEAR)
    heatmap_np = np.array(heatmap_img) / 255.0

    colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(heatmap_np)[:, :, :3]  # drop alpha channel
    colored_heatmap = np.uint8(colored_heatmap * 255)

    orig_np = np.array(orig_resized).astype(np.float32)
    overlay = (1 - alpha) * orig_np + alpha * colored_heatmap.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
