"""
Dataset loader expecting the standard Kaggle "Chest X-Ray Images (Pneumonia)"
directory layout (Kermany et al. dataset, widely used for this task):

    data/
      train/
        NORMAL/*.jpeg
        PNEUMONIA/*.jpeg
      val/
        NORMAL/*.jpeg
        PNEUMONIA/*.jpeg
      test/
        NORMAL/*.jpeg
        PNEUMONIA/*.jpeg

Download: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
(or via `kaggle datasets download -d paultimothymooney/chest-xray-pneumonia`)
"""
from torchvision.datasets import ImageFolder

from app.ml.preprocess import train_transform, val_transform


def get_datasets(data_root: str = "C:\\Users\\vigne\\Downloads\\medical ai\\medical-ai-platform\\medical-ai-platform\\training\\chest_xray"):
    train_ds = ImageFolder(f"{data_root}/train", transform=train_transform)
    val_ds = ImageFolder(f"{data_root}/val", transform=val_transform)
    test_ds = ImageFolder(f"{data_root}/test", transform=val_transform)

    # Sanity check: class_to_idx should match app/ml/model.CLASS_NAMES ordering
    assert train_ds.classes == ["NORMAL", "PNEUMONIA"], (
        f"Unexpected class ordering {train_ds.classes}; expected ['NORMAL', 'PNEUMONIA']. "
        "Check your data/train subfolder names."
    )

    return train_ds, val_ds, test_ds
