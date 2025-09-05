from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ImageFolderFlat(Dataset):
    """
    Simple flat image dataset:
    - root contains images directly OR
    - root contains subfolders; all images are used.
    Optionally returns (image, label) where label=0 for "good", 1 for "anomaly" if path contains "anomaly".
    """
    def __init__(self, root: str, img_size: int = 256, return_label: bool = False):
        self.root = Path(root)
        self.paths = [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]]
        self.img_size = img_size
        self.return_label = return_label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        im = Image.open(p).convert("RGB").resize((self.img_size, self.img_size))
        x = torch.from_numpy(np.array(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = x * 2 - 1  # normalize to [-1, 1]

        if self.return_label:
            # 'good' -> 0, tutto il resto -> 1
            label = 0 if p.parent.name.lower() == "good" else 1
            return x, label, str(p)
        return x, str(p)