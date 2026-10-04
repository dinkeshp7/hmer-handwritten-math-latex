"""
CROHME InkML Dataset Parser & 2D Anti-Aliased Rasterizer.

Parses XML-based InkML trace coordinates from CROHME datasets into 2D normalized
grayscale images and LaTeX ground-truth target strings.
"""

import xml.etree.ElementTree as ET
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Any


def parse_inkml(inkml_path: Union[str, Path]) -> Tuple[List[np.ndarray], str]:
    """
    Parses an InkML file and extracts trace coordinate arrays and LaTeX truth string.
    """
    path = Path(inkml_path)
    if not path.exists():
        raise FileNotFoundError(f"InkML file not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    # XML namespaces in InkML
    ns = {"inkml": "http://www.w3.org/2003/InkML"}

    # Extract LaTeX truth string
    truth = ""
    for annotation in root.findall(".//inkml:annotation", ns):
        if annotation.attrib.get("type") in ["truth", "normalized_truth"]:
            truth = annotation.text.strip() if annotation.text else ""
            break
            
    if not truth:
        # Fallback search without namespace
        for annotation in root.findall(".//annotation"):
            if annotation.attrib.get("type") in ["truth", "normalized_truth"]:
                truth = annotation.text.strip() if annotation.text else ""
                break

    # Extract traces
    traces = []
    trace_elements = root.findall(".//inkml:trace", ns) or root.findall(".//trace")
    
    for trace_elem in trace_elements:
        if not trace_elem.text:
            continue
        coords_raw = trace_elem.text.strip().split(",")
        coords = []
        for pt in coords_raw:
            parts = pt.strip().split()
            if len(parts) >= 2:
                coords.append([float(parts[0]), float(parts[1])])
        if len(coords) > 0:
            traces.append(np.array(coords, dtype=np.float32))

    return traces, truth


def rasterize_traces(
    traces: List[np.ndarray],
    target_size: Tuple[int, int] = (128, 256),
    padding: int = 10,
    stroke_width: int = 2
) -> np.ndarray:
    """
    Rasterizes trace coordinate arrays into a normalized 2D anti-aliased grayscale image.
    Output image dimensions: target_size = (Height, Width).
    """
    h_target, w_target = target_size
    canvas = np.zeros((h_target, w_target), dtype=np.uint8)

    if not traces:
        return canvas

    # Combine all trace points to find global bounding box
    all_pts = np.vstack(traces)
    x_min, y_min = np.min(all_pts[:, :2], axis=0)
    x_max, y_max = np.max(all_pts[:, :2], axis=0)

    w_box = max(x_max - x_min, 1.0)
    h_box = max(y_max - y_min, 1.0)

    # Compute scale factor preserving aspect ratio
    scale = min((h_target - 2 * padding) / h_box, (w_target - 2 * padding) / w_box)

    # Center alignment
    x_offset = (w_target - w_box * scale) / 2.0
    y_offset = (h_target - h_box * scale) / 2.0

    # Draw anti-aliased stroke lines
    for trace in traces:
        norm_trace = trace[:, :2].copy()
        norm_trace[:, 0] = (norm_trace[:, 0] - x_min) * scale + x_offset
        norm_trace[:, 1] = (norm_trace[:, 1] - y_min) * scale + y_offset

        pts = np.int32(norm_trace)
        for i in range(len(pts) - 1):
            pt1 = tuple(pts[i])
            pt2 = tuple(pts[i + 1])
            cv2.line(canvas, pt1, pt2, 255, stroke_width, cv2.LINE_AA)

    return canvas


if __name__ == "__main__":
    print("=== Testing CROHME InkML Rasterizer ===")
    # Generate synthetic trace coordinates representing x + 1
    trace1 = np.array([[10, 50], [30, 50], [20, 30], [20, 70]], dtype=np.float32)  # '+'
    trace2 = np.array([[50, 20], [50, 80]], dtype=np.float32)  # '1'
    synthetic_traces = [trace1, trace2]

    img = rasterize_traces(synthetic_traces, target_size=(128, 256))
    print(f"Rasterized Image Shape: {img.shape}")
    assert img.shape == (128, 256), "Rasterized dimensions mismatch target size!"
    print("[OK] CROHME InkML Rasterizer self-test passed cleanly!")
