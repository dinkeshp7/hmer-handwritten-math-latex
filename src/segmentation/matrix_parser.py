"""
2D Matrix Grid Segmenter & Cell Parser Engine for Handwritten Answer Scripts.

Segments multi-row, multi-column handwritten matrices (e.g. 4x4 matrix A in MA102 exam)
into individual row and cell crops, transcribing them into formatted LaTeX bmatrix code.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2


class MatrixGridParser:
    """
    Parses 2D handwritten matrix crops into LaTeX bmatrix environments.
    """
    def __init__(self, min_rows: int = 2, max_rows: int = 6):
        self.min_rows = min_rows
        self.max_rows = max_rows

    def segment_matrix_rows(self, matrix_crop: np.ndarray) -> List[np.ndarray]:
        """
        Segments a 2D matrix crop image into horizontal row crop images using horizontal projection.
        Excludes outer vertical bracket columns (left and right margins) to clearly expose inter-row valleys.
        """
        if matrix_crop.size == 0:
            return []

        h, w = matrix_crop.shape[:2]
        if matrix_crop.ndim == 3:
            gray = cv2.cvtColor(matrix_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = matrix_crop.copy()

        # Isolate interior matrix cells (excluding tall outer brackets)
        left_margin = int(w * 0.18)
        right_margin = int(w * 0.85)
        inner = gray[:, left_margin:right_margin] if (right_margin > left_margin) else gray

        binarized = cv2.adaptiveThreshold(
            inner, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 8
        )

        proj = np.sum(binarized > 0, axis=1)
        kernel_size = max(int(h * 0.05), 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = cv2.GaussianBlur(proj.astype(np.float32), (1, kernel_size), 0).flatten()

        # Find row valleys (local minima in interior projection)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(smoothed, distance=int(h * 0.15), prominence=max(np.mean(smoothed) * 0.2, 5))
        if len(peaks) >= self.min_rows:
            # Valleys between peaks
            valleys = [0]
            for i in range(len(peaks) - 1):
                p1, p2 = peaks[i], peaks[i+1]
                valley_idx = p1 + int(np.argmin(smoothed[p1:p2]))
                valleys.append(valley_idx)
            valleys.append(h)

            row_crops = [gray[valleys[i]:valleys[i+1], :] for i in range(len(valleys)-1)]
            return row_crops

        threshold = np.mean(smoothed) * 0.25
        row_mask = smoothed > threshold

        row_intervals: List[Tuple[int, int]] = []
        in_row = False
        start_y = 0

        for y in range(h):
            if row_mask[y] and not in_row:
                in_row = True
                start_y = y
            elif not row_mask[y] and in_row:
                in_row = False
                if (y - start_y) >= 12:
                    row_intervals.append((start_y, y))

        if in_row and (h - start_y) >= 12:
            row_intervals.append((start_y, h))

        row_crops = [gray[y1:y2, :] for y1, y2 in row_intervals]
        return row_crops if len(row_crops) >= self.min_rows else [gray]

    def parse_matrix_to_latex(self, matrix_crop: np.ndarray, cell_recognizer=None) -> str:
        """
        Parses handwritten matrix crop into LaTeX bmatrix code by segmenting row crops.
        """
        rows = self.segment_matrix_rows(matrix_crop)
        if not rows:
            return r"\begin{bmatrix} 0 \end{bmatrix}"

        latex_rows = []
        for idx, row in enumerate(rows):
            if cell_recognizer is not None:
                row_str = cell_recognizer(row)
            else:
                row_str = f"r_{{{idx+1}1}} & r_{{{idx+1}2}} & r_{{{idx+1}3}} & r_{{{idx+1}4}}"
            if row_str:
                latex_rows.append(row_str)

        if not latex_rows:
            return r"\begin{bmatrix} 0 \end{bmatrix}"

        body = " \\\\\n".join(latex_rows)
        return f"\\begin{{bmatrix}}\n{body}\n\\end{{bmatrix}}"


if __name__ == "__main__":
    print("=== Testing Matrix Grid Parser Engine ===")
    parser = MatrixGridParser()
    dummy_matrix = np.full((200, 300), 255, dtype=np.uint8)
    # Draw 4 dummy rows
    for y in [30, 80, 130, 180]:
        cv2.line(dummy_matrix, (20, y), (280, y), 0, 4)

    rows = parser.segment_matrix_rows(dummy_matrix)
    print(f"Segmented Matrix Rows: {len(rows)}")
    latex_code = parser.parse_matrix_to_latex(dummy_matrix)
    print("\nGenerated LaTeX bmatrix:")
    print(latex_code)
    assert len(rows) >= 2, "Matrix row segmentation failed!"
    print("[OK] Matrix Grid Parser self-test passed cleanly!")
