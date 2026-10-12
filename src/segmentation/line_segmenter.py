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
        Measures if any single connected stroke component spans across both
        the vertical (> 140px) and horizontal (> 350px) extent of the region.
        """
        if crop_mask.size == 0:
            return False
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(crop_mask)
        for s in stats[1:]:
            cw = s[cv2.CC_STAT_WIDTH]
            ch = s[cv2.CC_STAT_HEIGHT]
            # A continuous slash/cross-out stroke spanning across lines
            if ch > 140 and cw > 350:
                return True

        return False

    def classify_region_type(self, w: int, h: int, crop: np.ndarray) -> str:
        """
        Classifies region into 'matrix', 'math_block', or 'text_line' based on aspect ratio and geometry.
        """
        aspect_ratio = w / float(max(h, 1))

        # True 2D matrices are tall, multi-row structures (height > 150, aspect ratio 0.8 to 5.2)
        # Full-page single lines of text or math have aspect ratio > 6.0.
        if h > 150 and 0.8 <= aspect_ratio <= 5.2:
            return "matrix"
        elif h < 65 and aspect_ratio < 4.0:
            return "math_block"
        else:
            return "text_line"

    def deduplicate_bounding_boxes(self, boxes: List[Tuple[int, int, int, int]], iou_threshold: float = 0.45) -> List[Tuple[int, int, int, int]]:
        """
        Groups bounding box fragments into cohesive full text lines and 2D matrix blocks.
        """
        if not boxes:
            return []

        sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        merged: List[Tuple[int, int, int, int]] = []

        for box in sorted_boxes:
            x1, y1, w1, h1 = box
            yc1 = y1 + h1 / 2.0
            matched = False

            for idx, (mx, my, mw, mh) in enumerate(merged):
                yc_m = my + mh / 2.0
                v_overlap = max(0, min(y1 + h1, my + mh) - max(y1, my))
                min_h = min(h1, mh)

                # Case A: Same horizontal text line (h < 110, mh < 110, vertical centers close)
                same_line = (h1 < 110 and mh < 110 and abs(yc1 - yc_m) < 35)

                # Case B: Part of the same tall 2D block / matrix (vertical overlap > 60% of min_h)
                same_block = (min_h > 0 and (v_overlap / float(min_h)) > 0.65 and max(h1, mh) > 130 and abs(yc1 - yc_m) < 120)

                if same_line or same_block:
                    new_x = min(x1, mx)
                    new_y = min(y1, my)
                    new_w = max(x1 + w1, mx + mw) - new_x
                    new_h = max(y1 + h1, my + mh) - new_y
                    merged[idx] = (new_x, new_y, new_w, new_h)
                    matched = True
                    break

            if not matched:
                merged.append(box)

        # Final sort top-to-bottom
        return sorted(merged, key=lambda b: b[1])

    def segment_page(self, image_gray: np.ndarray, binarized_mask: np.ndarray) -> List[RegionCrop]:
        """
        Segments a full page grayscale image into an ordered list of RegionCrop objects.
        """
        h_page, w_page = image_gray.shape[:2]

        # 1. Pure Horizontal Dilation for Line Segmentation (prevents vertical bleed)
        kernel_w = max(int(w_page * 0.035), 25)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        dilated = cv2.dilate(binarized_mask, kernel, iterations=1)

        # 2. Find External Bounding Contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_regions: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter printed header margin instructions & question prompt box (top 26.5% of page)
            if y < int(h_page * 0.265):
                continue
            # Filter bottom margin artifacts (bottom 5% of page)
            if y > int(h_page * 0.95):
                continue
            # Filter thin horizontal printed separator lines
            if w > int(w_page * 0.55) and h < 22:
                continue
            if h < self.min_line_height or w < 30:
                continue

            raw_regions.append((x, y, w, h))

        # 3. Deduplicate and cluster bounding boxes into cohesive lines and 2D blocks
        dedup_regions = self.deduplicate_bounding_boxes(raw_regions, iou_threshold=0.45)

        # 4. Extract Crop Tensors and Classify
        regions: List[RegionCrop] = []
        for idx, (x, y, w, h) in enumerate(dedup_regions, 1):
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
    dummy_page = np.full((1000, 600), 255, dtype=np.uint8)
    cv2.putText(dummy_page, "Given S in M_4x4(R)", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.rectangle(dummy_page, (50, 380), (400, 580), 0, 2)  # Matrix block
    cv2.putText(dummy_page, "To show S is subspace", (50, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    binarized = cv2.adaptiveThreshold(dummy_page, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    regions = segmenter.segment_page(dummy_page, binarized)

    print(f"Total Regions Segmented: {len(regions)}")
    for r in regions:
        print(f"  Region #{r.crop_id}: bbox={r.bbox}, type={r.region_type}, crop_shape={r.image_crop.shape}")

    assert len(regions) >= 2, "Line segmenter failed to extract regions!"
    print("[OK] Page Line Segmenter self-test passed cleanly!")
