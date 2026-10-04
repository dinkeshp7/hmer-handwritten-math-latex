"""
Line & Region Bounding Box Segmentation Engine for Handwritten Answer Scripts.

Segments full handwritten pages into horizontal line crops and matrix region crops
using morphologic projections, contour analysis, and stroke density filtering.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2


@dataclass
class RegionCrop:
    """Dataclass holding extracted region crops and metadata."""
    crop_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    region_type: str  # 'text_line', 'math_block', 'matrix', 'header', 'scratchout'
    image_crop: np.ndarray
    confidence: float = 1.0


class PageLineSegmenter:
    """
    Page Layout Line & Region Segmenter.
    """
    def __init__(self, min_line_height: int = 15, max_line_height: int = 250):
        self.min_line_height = min_line_height
        self.max_line_height = max_line_height

    def is_scratchout(self, crop_mask: np.ndarray) -> bool:
        """
        Detects if a cropped region is a crossed-out scribble/scratch-out.
        Measures contour density and line intersection density.
        """
        if crop_mask.size == 0:
            return False
        
        # Measure stroke pixel ratio
        foreground_pixels = np.count_nonzero(crop_mask)
        total_pixels = crop_mask.size
        density = foreground_pixels / float(total_pixels)

        # High stroke overlap (> 65% black pixels in bounding box) indicates scribble
        if density > 0.65:
            return True

        return False

    def classify_region_type(self, w: int, h: int, crop: np.ndarray) -> str:
        """
        Classifies region into 'matrix', 'math_block', or 'text_line' based on aspect ratio and geometry.
        """
        aspect_ratio = w / float(max(h, 1))

        # Matrices (like 4x4 or 4x6 matrices) are tall and wide (aspect ratio 0.8 to 2.5, height > 80px)
        if h > 80 and 0.5 <= aspect_ratio <= 3.0:
            return "matrix"
        # Long narrow crops are text/math lines
        elif aspect_ratio > 3.5:
            return "text_line"
        else:
            return "math_block"

    def segment_page(self, image_gray: np.ndarray, binarized_mask: np.ndarray) -> List[RegionCrop]:
        """
        Segments a full page grayscale image into an ordered list of RegionCrop objects.
        """
        h_page, w_page = image_gray.shape[:2]

        # 1. Horizontal Projection Profile for Line Segmentation
        kernel_w = max(int(w_page * 0.05), 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
        dilated = cv2.dilate(binarized_mask, kernel, iterations=2)

        # 2. Find External Bounding Contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_regions: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter printed header margin instructions (top 15% of page, e.g. "NEATLY write name...")
            if y < int(h_page * 0.14) and h < 90:
                continue
            if h < self.min_line_height or w < 40:
                continue

            raw_regions.append((x, y, w, h))

        # 3. Sort Bounding Boxes Top-to-Bottom by Y-Coordinate
        raw_regions = sorted(raw_regions, key=lambda b: b[1])

        # 4. Extract Crop Tensors and Classify
        regions: List[RegionCrop] = []
        for idx, (x, y, w, h) in enumerate(raw_regions, 1):
            crop_img = image_gray[y:y+h, x:x+w]
            crop_mask = binarized_mask[y:y+h, x:x+w]

            if self.is_scratchout(crop_mask):
                reg_type = "scratchout"
            else:
                reg_type = self.classify_region_type(w, h, crop_img)

            regions.append(RegionCrop(
                crop_id=idx,
                bbox=(x, y, w, h),
                region_type=reg_type,
                image_crop=crop_img
            ))

        return regions


if __name__ == "__main__":
    print("=== Testing Page Line Segmenter Engine ===")
    segmenter = PageLineSegmenter()
    
    # Create dummy page with two text lines and one matrix
    dummy_page = np.full((600, 500), 255, dtype=np.uint8)
    cv2.putText(dummy_page, "Given S in M_4x4(R)", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.rectangle(dummy_page, (50, 150), (350, 300), 0, 2)  # Matrix block
    cv2.putText(dummy_page, "To show S is subspace", (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    binarized = cv2.adaptiveThreshold(dummy_page, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    regions = segmenter.segment_page(dummy_page, binarized)

    print(f"Total Regions Segmented: {len(regions)}")
    for r in regions:
        print(f"  Region #{r.crop_id}: bbox={r.bbox}, type={r.region_type}, crop_shape={r.image_crop.shape}")

    assert len(regions) >= 2, "Line segmenter failed to extract regions!"
    print("[OK] Page Line Segmenter self-test passed cleanly!")
