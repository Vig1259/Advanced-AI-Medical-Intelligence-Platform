"""
Standalone evaluation of a trained checkpoint on the test set, with a
saved confusion matrix and ROC curve for the project report.

Usage:
    python training/evaluate.py --checkpoint models/chest_xray_densenet121.pt --data-root data
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import RocCurveDisplay, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.model import build_model, load_checkpoint  # noqa: E402
from training.dataset import get_datasets  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/chest_xray_densenet121.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="docs/eval_plots")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    _, _, test_ds = get_datasets(args.data_root)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    model = build_model(pretrained=False)
    model = load_checkpoint(model, args.checkpoint, device=device)

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(all_labels, all_probs)
    plt.title("ROC Curve - Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "roc_curve.png"), dpi=150)
    plt.close()

    print(f"Saved evaluation plots to {args.output_dir}")


if __name__ == "__main__":
    main()
