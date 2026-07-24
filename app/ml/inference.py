"""High-level inference service: loads the model once and exposes predict()."""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass

import torch
from PIL import Image

from app.ml.gradcam import GradCAM, image_to_png_bytes, overlay_heatmap_on_image
from app.ml.model import CLASS_NAMES, build_model, load_checkpoint
from app.ml.preprocess import preprocess_for_inference


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    class_probabilities: dict
    gradcam_base64_png: str


class InferenceService:
    """Singleton-style service. Instantiate once at app startup."""

    def __init__(self, checkpoint_path: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(pretrained=(checkpoint_path is None))

        checkpoint_path = checkpoint_path or os.environ.get("MODEL_CHECKPOINT_PATH", "models/chest_xray_densenet121.pt")
        if os.path.exists(checkpoint_path):
            self.model = load_checkpoint(self.model, checkpoint_path, device=self.device)
        else:
            # Falls back to ImageNet-pretrained backbone with an untrained head.
            # Predictions will be meaningless until a trained checkpoint is provided --
            # this keeps the API usable for wiring/testing purposes.
            self.model.to(self.device)
            self.model.eval()

        self.gradcam = GradCAM(self.model)

    def predict(self, image: Image.Image) -> PredictionResult:
        input_tensor = preprocess_for_inference(image).to(self.device)
        input_tensor.requires_grad_(True)

        heatmap, pred_idx, confidence = self.gradcam.generate(input_tensor)

        # Recompute full probability vector (gradcam.generate already ran a forward pass,
        # but we re-run without grad tracking for a clean read of all class probs)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

        class_probabilities = {CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(probs)}

        overlay = overlay_heatmap_on_image(image, heatmap)
        png_bytes = image_to_png_bytes(overlay)
        gradcam_b64 = base64.b64encode(png_bytes).decode("utf-8")

        return PredictionResult(
            predicted_class=CLASS_NAMES[pred_idx],
            confidence=round(confidence, 4),
            class_probabilities=class_probabilities,
            gradcam_base64_png=gradcam_b64,
        )


_service_instance: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _service_instance
    if _service_instance is None:
        _service_instance = InferenceService()
    return _service_instance
