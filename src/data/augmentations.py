"""
Handwritten Image Augmentations for HMER Training.

Provides realistic handwriting augmentations including:
1. Elastic distortion (simulating muscle tremors & paper warp)
2. Random Affine transformations (rotation, shear, scaling)
3. Morphological dilation/erosion (simulating ballpoint pen vs thick marker ink)
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def elastic_distortion(
    image: np.ndarray,
    alpha: float = 36.0,
    sigma: float = 6.0,
    random_state: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Applies elastic distortion to a 2D grayscale image array (H, W).
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = cv2.GaussianBlur(
        (random_state.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma
    ) * alpha
    dy = cv2.GaussianBlur(
        (random_state.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma
    ) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def random_affine_transform(
    image: np.ndarray,
    max_rotation_deg: float = 5.0,
    max_shear_deg: float = 8.0,
    scale_range: Tuple[float, float] = (0.95, 1.05)
) -> np.ndarray:
    """
    Applies random rotation, shear, and scale affine transformations.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Random parameters
    angle = np.random.uniform(-max_rotation_deg, max_rotation_deg)
    shear = np.random.uniform(-max_shear_deg, max_shear_deg)
    scale = np.random.uniform(scale_range[0], scale_range[1])

    # Rotation + Scale Matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Apply Shear
    shear_rad = np.radians(shear)
    M[0, 1] += np.tan(shear_rad)

    transformed = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    return transformed


def stroke_morphology_augmentation(image: np.ndarray) -> np.ndarray:
    """
    Randomly applies morphological erosion (thin ink) or dilation (thick ink).
    """
    kernel_size = np.random.choice([1, 2, 3])
    if kernel_size == 1:
        return image

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    op = np.random.choice(["erode", "dilate", "none"])

    if op == "erode":
        return cv2.erode(image, kernel, iterations=1)
    elif op == "dilate":
        return cv2.dilate(image, kernel, iterations=1)
    else:
        return image


def apply_hmer_augmentations(image: np.ndarray) -> np.ndarray:
    """
    Pipeline applying a sequence of realistic handwritten augmentations.
    Expects grayscale uint8 image array (H, W).
    """
    aug_img = image.copy()
    
    # 1. Random Affine (50% probability)
    if np.random.rand() > 0.5:
        aug_img = random_affine_transform(aug_img)

    # 2. Random Elastic Distortion (30% probability)
    if np.random.rand() > 0.7:
        aug_img = elastic_distortion(aug_img, alpha=20.0, sigma=4.0)

    # 3. Morphological Stroke Modification (40% probability)
    if np.random.rand() > 0.6:
        aug_img = stroke_morphology_augmentation(aug_img)

    return aug_img


if __name__ == "__main__":
    print("=== Testing HMER Image Augmentations ===")
    # Create a dummy synthetic binary image with a diagonal stroke
    dummy_img = np.zeros((128, 256), dtype=np.uint8)
    cv2.line(dummy_img, (20, 20), (200, 100), 255, 3)

    augmented = apply_hmer_augmentations(dummy_img)
    print(f"Original Shape: {dummy_img.shape}, Augmented Shape: {augmented.shape}")
    assert dummy_img.shape == augmented.shape, "Augmentation modified image dimensions!"
    print("[OK] HMER Image Augmentations self-test passed cleanly!")
