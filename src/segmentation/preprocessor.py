"""
Image Preprocessing Engine for Handwritten Answer Scripts (IIT Guwahati MA102 Dataset).

Applies contrast enhancement (CLAHE), adaptive binarization (Sauvola/Otsu),
and deskewing to prepare full-page scanned/mobile photos for segmentation.
"""

import math
from typing import Tuple, Optional
import numpy as np
import cv2


class ImagePreprocessor:
    """
    Robust image preprocessing pipeline for handwritten document pages.
    """
    def __init__(self, target_dpi: int = 300, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.target_dpi = target_dpi
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance_contrast(self, image_gray: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        return self.clahe.apply(image_gray)

    def estimate_skew_angle(self, image_gray: np.ndarray) -> float:
        """Estimates document skew angle in degrees using Hough line transformation."""
        edges = cv2.Canny(image_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if -45 < angle < 45:
                angles.append(angle)

        return float(np.median(angles)) if len(angles) > 0 else 0.0

    def deskew(self, image_gray: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """Deskews image to correct vertical alignment."""
        if angle is None:
            angle = self.estimate_skew_angle(image_gray)

        if abs(angle) < 0.5:
            return image_gray

        h, w = image_gray.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(
            image_gray, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return deskewed

    def process_page(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Master preprocessing pipeline for a full handwritten page image.
        Returns: (Enhanced Grayscale Image, Binarized Mask)
        """
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)
            elif image.shape[2] == 1:
                image = image.squeeze(2)

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Enhance Contrast (CLAHE)
        enhanced = self.enhance_contrast(gray)

        # 2. Deskew Alignment
        deskewed = self.deskew(enhanced)

        # 3. Adaptive Threshold Binarization
        binarized = cv2.adaptiveThreshold(
            deskewed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )

        return deskewed, binarized


if __name__ == "__main__":
    print("=== Testing Image Preprocessor Engine ===")
    preprocessor = ImagePreprocessor()
    dummy_img = np.full((500, 400), 200, dtype=np.uint8)
    cv2.putText(dummy_img, "Test Answer Script Page", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 50, 2)

    enhanced, binarized = preprocessor.process_page(dummy_img)
    print(f"Enhanced Shape: {enhanced.shape}, Binarized Shape: {binarized.shape}")
    assert enhanced.shape == dummy_img.shape, "Enhanced shape mismatch!"
    print("[OK] Image Preprocessor self-test passed cleanly!")
