"""Image preprocessing for chest X-ray inference."""
from __future__ import annotations

import io

import torch
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224

# Standard ImageNet normalization (matches DenseNet121 pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),  # X-rays are grayscale; expand to 3ch for pretrained net
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

val_transform = inference_transform


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def preprocess_for_inference(image: Image.Image) -> torch.Tensor:
    """Returns a (1, 3, H, W) tensor ready for the model."""
    tensor = inference_transform(image)
    return tensor.unsqueeze(0)
