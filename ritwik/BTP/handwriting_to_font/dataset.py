"""
Paired dataset for Stage 1: (handwriting line crop, font line crop).

Expects a directory structure:
    root/
      handwriting/  0001.png  0002.png  ...
      font/         0001.png  0002.png  ...
matched by filename. Crops are variable width -- padding + masking is
handled in the collate function so they can be batched together.
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PairedLineDataset(Dataset):
    def __init__(self, root, height=64, max_width=768):
        self.root = Path(root)
        self.height = height
        self.max_width = max_width
        hw_dir = self.root / "handwriting"
        self.filenames = sorted(f.name for f in hw_dir.glob("*.png"))
        self.to_tensor = transforms.ToTensor()  # -> [0, 1]

    def __len__(self):
        return len(self.filenames)

    def _load(self, subdir, name):
        img = Image.open(self.root / subdir / name).convert("L")  # grayscale
        w, h = img.size
        new_w = max(1, round(w * self.height / h))
        new_w = min(new_w, self.max_width)
        img = img.resize((new_w, self.height), Image.BILINEAR)
        tensor = self.to_tensor(img)  # (1, H, W) in [0, 1]
        return tensor * 2 - 1          # -> [-1, 1], matches generator's Tanh output

    def __getitem__(self, idx):
        name = self.filenames[idx]
        handwriting = self._load("handwriting", name)
        font = self._load("font", name)
        return handwriting, font


def collate_pad(batch):
    """
    Pads all crops in a batch to the widest crop, rounded up to a multiple
    of 16 (required by the U-Net's 4 downsampling stages). Returns a mask
    marking real (1) vs padded (0) columns -- apply it before computing any
    loss so padding never leaks into the training signal.
    """
    heights = {t.shape[1] for pair in batch for t in pair}
    assert len(heights) == 1, "all crops must share the same height"
    height = heights.pop()

    max_w = max(t.shape[2] for pair in batch for t in pair)
    max_w = ((max_w + 15) // 16) * 16  # round up to multiple of 16

    hw_batch, font_batch, mask_batch = [], [], []
    for handwriting, font in batch:
        w = handwriting.shape[2]
        pad = max_w - w

        hw_padded = torch.nn.functional.pad(handwriting, (0, pad), value=-1.0)
        font_padded = torch.nn.functional.pad(font, (0, pad), value=-1.0)
        mask = torch.zeros(1, height, max_w)
        mask[:, :, :w] = 1.0

        hw_batch.append(hw_padded)
        font_batch.append(font_padded)
        mask_batch.append(mask)

    return torch.stack(hw_batch), torch.stack(font_batch), torch.stack(mask_batch)
