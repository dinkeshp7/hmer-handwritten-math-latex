"""
Dataset for Stage 2: (font-rendered page image, LaTeX source[, line boxes]).

Expects a directory structure:
    root/
      images/  0001.png   0002.png  ...
      latex/   0001.tex   0002.tex  ...   (matching filenames)
      boxes/   0001.json  0002.json ...   (optional: list of [x0,y0,x1,y1], normalized 0-1)
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class Img2LatexDataset(Dataset):
    def __init__(self, root, tokenizer, patch_size=16, max_len=1024, use_boxes=False):
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.max_len = max_len
        self.use_boxes = use_boxes
        self.filenames = sorted(f.stem for f in (self.root / "images").glob("*.png"))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        img = Image.open(self.root / "images" / f"{name}.png").convert("L")
        w, h = img.size
        # pad to a multiple of patch_size so patch embedding divides evenly
        new_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
        padded = Image.new("L", (new_w, new_h), color=255)
        padded.paste(img, (0, 0))
        image_tensor = self.to_tensor(padded) * 2 - 1  # -> [-1, 1]

        latex_str = (self.root / "latex" / f"{name}.tex").read_text(encoding="utf-8")
        ids = self.tokenizer.encode(latex_str)[: self.max_len]
        target_ids = torch.tensor(ids, dtype=torch.long)

        boxes = None
        if self.use_boxes:
            box_path = self.root / "boxes" / f"{name}.json"
            raw_boxes = json.loads(box_path.read_text())
            boxes = torch.tensor(raw_boxes, dtype=torch.float)

        return image_tensor, target_ids, boxes


def collate_fn(batch, pad_idx=0):
    images, targets, boxes_list = zip(*batch)

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    padded_images = []
    for img in images:
        pad_h, pad_w = max_h - img.shape[1], max_w - img.shape[2]
        padded_images.append(torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=-1.0))
    images_batch = torch.stack(padded_images)

    max_len = max(t.shape[0] for t in targets)
    padded_targets = []
    for t in targets:
        pad_len = max_len - t.shape[0]
        padded_targets.append(torch.nn.functional.pad(t, (0, pad_len), value=pad_idx))
    targets_batch = torch.stack(padded_targets)

    boxes_batch, box_mask_batch = None, None
    if boxes_list[0] is not None:
        max_boxes = max(b.shape[0] for b in boxes_list)
        padded_boxes, masks = [], []
        for b in boxes_list:
            pad_n = max_boxes - b.shape[0]
            padded_boxes.append(torch.nn.functional.pad(b, (0, 0, 0, pad_n), value=0.0))
            mask = torch.zeros(max_boxes)
            mask[: b.shape[0]] = 1.0
            masks.append(mask)
        boxes_batch = torch.stack(padded_boxes)
        box_mask_batch = torch.stack(masks)

    return images_batch, targets_batch, boxes_batch, box_mask_batch
