"""
PyTorch Dataset definition for fingernail photographs with patient ID tracking.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class NailDataset(Dataset):
    """
    Dataset wrapper for nail photographs and binary anemia labels.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(float(row["anemia_label"]), dtype=torch.float32)
        patient_id = row.get("patient_id", f"patient_{idx}")
        image_id = row.get("image_id", f"image_{idx}")

        return image, label, patient_id, image_id
