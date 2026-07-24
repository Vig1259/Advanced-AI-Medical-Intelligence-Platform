"""
Training script for the chest X-ray pneumonia classifier.

Usage:
    python training/train.py --data-root data --epochs 15 --batch-size 32 --lr 1e-4

Requires the Kaggle "Chest X-Ray Images (Pneumonia)" dataset laid out as
described in training/dataset.py. This script was NOT executed in the
assignment sandbox (no GPU/network access there) -- run it in your own
environment (local GPU, Colab, or Kaggle notebook).
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.model import build_model  # noqa: E402
from training.dataset import get_datasets  # noqa: E402


def compute_class_weights(dataset) -> torch.Tensor:
    """Handles class imbalance (the Kaggle dataset has ~3x more PNEUMONIA than NORMAL)."""
    counts = [0, 0]
    for _, label in dataset.samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (2 * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    report = classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"])
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, auc, report, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="models/chest_xray_densenet121.pt")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2,
                         help="Number of initial epochs training only the classifier head")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = get_datasets(args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_ds).to(device)
    print(f"Class weights (NORMAL, PNEUMONIA): {class_weights.tolist()}")

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_auc = -1.0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for epoch in range(args.epochs):
        # Optionally freeze the backbone for the first few epochs (head-only warmup)
        for p in model.features.parameters():
            p.requires_grad = epoch >= args.freeze_backbone_epochs

        model.train()
        start = time.time()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f} "
            f"({elapsed:.1f}s)"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {"model_state_dict": model.state_dict(), "val_auc": val_auc, "epoch": epoch},
                args.output,
            )
            print(f"  -> saved new best checkpoint (val_auc={val_auc:.4f}) to {args.output}")

    # Final test-set evaluation using the best checkpoint
    print("\nLoading best checkpoint for final test evaluation...")
    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_auc, report, cm = evaluate(model, test_loader, device, criterion)

    print(f"\nTest results: loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f}")
    print("\nClassification report:\n", report)
    print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()
