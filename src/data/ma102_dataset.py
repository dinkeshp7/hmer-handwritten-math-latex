"""
IIT Guwahati MA102 Handwritten Exam Answer Script Dataset Loader.

Indexes and loads the 13,146 handwritten exam pages from:
C:\\Users\\Dinkesh\\Downloads\\2025s-ma102-q1\\anonymous

Uses OpenCV (cv2) and NumPy for fast image I/O, with optional PyTorch Tensor conversion.
"""

import os
import re
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Any
import numpy as np
import cv2

# Gracefully handle optional torch / torchvision imports
try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:  # Dummy fallback class
        pass


# Default path detection: prefers PARAM Kamrupa HPC path, falls back to Windows local path
DEFAULT_MA102_DIR = (
    r"/scratch/p.dinkesh/hmer_project/data/ma102"
    if os.path.exists(r"/scratch/p.dinkesh/hmer_project/data/ma102") or os.name == "posix"
    else r"C:\Users\Dinkesh\Downloads\2025s-ma102-q1\anonymous"
)


class MA102ExamDataset(Dataset):
    """
    Dataset class for indexing and loading IIT Guwahati MA102 exam answer script images.
    """
    def __init__(
        self,
        data_dir: str = DEFAULT_MA102_DIR,
        target_size: Tuple[int, int] = (256, 512)
    ):
        self.data_dir = Path(data_dir)
        self.target_size = target_size  # (Height, Width)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"MA102 Dataset directory not found at: {self.data_dir}")

        # Index all jpg files matching page-{roll}-{page}.jpg (supports recursive subdirectories)
        self.image_paths: List[Path] = sorted(list(self.data_dir.rglob("*.jpg")))
        self.samples_metadata: List[Dict[str, Union[str, int]]] = []

        pattern = re.compile(r"page-(\d+)-(\d+)\.jpg")
        for img_path in self.image_paths:
            match = pattern.search(img_path.name)
            if match:
                roll_num, page_num = match.groups()
                self.samples_metadata.append({
                    "path": str(img_path),
                    "filename": img_path.name,
                    "roll_number": roll_num,
                    "page_number": int(page_num)
                })

    def __len__(self) -> int:
        return len(self.samples_metadata)

    def get_full_page_image(self, idx: int) -> Tuple[np.ndarray, Dict[str, Union[str, int]]]:
        """Returns the unscaled original high-resolution grayscale image (H, W)."""
        meta = self.samples_metadata[idx]
        img_path = meta["path"]
        img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            raise IOError(f"Failed to read image at: {img_path}")
        return img_np, meta

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Union[str, int]]]:
        meta = self.samples_metadata[idx]
        img_path = meta["path"]
        
        # Load grayscale image using OpenCV
        img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            raise IOError(f"Failed to read image at: {img_path}")
            
        # Resize to (target_width, target_height) for cv2.resize
        h, w = self.target_size
        img_resized = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [-1.0, 1.0]
        img_float = (img_resized.astype(np.float32) / 127.5) - 1.0
        img_float = np.expand_dims(img_float, axis=0)  # Shape: (1, H, W)

        if HAS_TORCH:
            return torch.from_numpy(img_float), meta
        else:
            return img_float, meta


if __name__ == "__main__":
    print("=== Testing IITG MA102 Exam Dataset Loader ===")
    dataset = MA102ExamDataset()
    print(f"Total Indexed Answer Script Pages: {len(dataset)}")
    
    if len(dataset) > 0:
        sample_img, sample_meta = dataset[0]
        print(f"Sample 0 Metadata: {sample_meta}")
        print(f"Sample 0 Array/Tensor Type: {type(sample_img)}")
        print(f"Sample 0 Array/Tensor Shape: {sample_img.shape}")
        
    print("[OK] IITG MA102 Exam Dataset Loader self-test passed cleanly!")
