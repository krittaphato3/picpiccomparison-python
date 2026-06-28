"""
Image loading, resizing, and matrix conversion module.

Handles the preprocessing pipeline: loading images from disk,
converting to grayscale, resizing to matching dimensions, and
flattening to vectors for linear algebra operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or is invalid."""


class DimensionMismatchError(Exception):
    """Raised when images cannot be resized to matching dimensions."""


@dataclass
class ImagePair:
    """Container for a pair of loaded and preprocessed images.

    Attributes:
        image_a: Grayscale image matrix A (H x W), float64 normalized [0, 1].
        image_b: Grayscale image matrix B (H x W), float64 normalized [0, 1].
        vector_a: Flattened vector of image A, shape (N,).
        vector_b: Flattened vector of image B, shape (N,).
        height: Common height of both images.
        width: Common width of both images.
        path_a: Original file path of image A.
        path_b: Original file path of image B.
    """

    image_a: np.ndarray
    image_b: np.ndarray
    vector_a: np.ndarray
    vector_b: np.ndarray
    height: int
    width: int
    path_a: str
    path_b: str


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk and convert to grayscale float64 matrix.

    Args:
        path: File system path to the image.

    Returns:
        Grayscale image as a 2D numpy array with dtype float64,
        values normalized to [0.0, 1.0].

    Raises:
        ImageLoadError: If the file does not exist or cannot be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"Image file not found: {path}")
    if not path.is_file():
        raise ImageLoadError(f"Path is not a file: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageLoadError(f"Failed to decode image: {path}")

    # Convert to grayscale if color (3 or 4 channels)
    if img.ndim == 3:
        if img.shape[2] == 4:
            # Drop alpha channel, keep BGR
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize to [0, 1] float64
    img = img.astype(np.float64) / 255.0
    return img


def resize_to_match(
    img_a: np.ndarray,
    img_b: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize two images to matching dimensions.

    If target_size is not provided, uses the dimensions of img_a as target.
    Both images are resized to the same (width, height).

    Args:
        img_a: First image matrix.
        img_b: Second image matrix.
        target_size: Optional (width, height) tuple. If None, uses img_a's size.

    Returns:
        Tuple of (resized_img_a, resized_img_b) with matching dimensions.
    """
    if target_size is None:
        target_size = (img_a.shape[1], img_a.shape[0])  # (width, height)

    # cv2.resize takes (width, height)
    if img_a.shape[:2] != (target_size[1], target_size[0]):
        img_a = cv2.resize(
            img_a, target_size, interpolation=cv2.INTER_AREA
        )
    if img_b.shape[:2] != (target_size[1], target_size[0]):
        img_b = cv2.resize(
            img_b, target_size, interpolation=cv2.INTER_AREA
        )

    return img_a, img_b


def flatten_image(img: np.ndarray) -> np.ndarray:
    """Flatten a 2D image matrix into a 1D vector.

    Args:
        img: 2D numpy array (H x W).

    Returns:
        1D numpy array of shape (N,) where N = H * W.
    """
    return img.flatten()


# ---------------------------------------------------------------------------
# Foreground masking and cropping (extracted from gui_similarity.py)
# ---------------------------------------------------------------------------

def auto_foreground_mask(
    img: np.ndarray, threshold: float = 0.95
) -> np.ndarray:
    """Compute a foreground mask by intensity thresholding.

    Pixels darker than the threshold are considered foreground.
    This is useful for isolating objects from bright/white backgrounds.

    Args:
        img: 2D grayscale image matrix, float64 in [0, 1].
        threshold: Intensity threshold. Pixels < threshold are foreground.
                   Default 0.95 (assumes white background).

    Returns:
        Boolean mask of shape (H, W), True where foreground.
    """
    return img < float(threshold)


def bbox_from_mask(
    mask: np.ndarray, margin: int = 2
) -> Tuple[int, int, int, int]:
    """Compute the bounding box from a boolean mask.

    Args:
        mask: Boolean mask of shape (H, W).
        margin: Pixel margin to add around the bounding box.

    Returns:
        Tuple (x0, y0, x1, y1) bounding box coordinates.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    y0 = max(0, int(ys.min() - margin))
    y1 = min(mask.shape[0], int(ys.max() + 1 + margin))
    x0 = max(0, int(xs.min() - margin))
    x1 = min(mask.shape[1], int(xs.max() + 1 + margin))
    return (x0, y0, x1, y1)


def crop_to_foreground(
    img: np.ndarray, mask: np.ndarray, margin: int = 2
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop an image to its foreground bounding box.

    Args:
        img: 2D image matrix.
        mask: Boolean foreground mask.
        margin: Pixel margin around the bounding box.

    Returns:
        Tuple of (cropped_image, (x0, y0, x1, y1) bounding box).
    """
    bbox = bbox_from_mask(mask, margin=margin)
    x0, y0, x1, y1 = bbox
    return img[y0:y1, x0:x1], bbox


def load_and_prepare_images(
    path_a: Union[str, Path],
    path_b: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
) -> ImagePair:
    """Load, preprocess, and validate a pair of images for comparison.

    This is the main entry point for the loader module. It loads both images,
    converts to grayscale, resizes to matching dimensions, normalizes, and
    flattens into vectors.

    Args:
        path_a: File path to the first image.
        path_b: File path to the second image.
        target_size: Optional (width, height) to resize both images to.
                     If None, resizes image B to match image A's dimensions.

    Returns:
        An ImagePair dataclass containing all preprocessed data.

    Raises:
        ImageLoadError: If either image cannot be loaded.
        DimensionMismatchError: If resized dimensions don't match.
    """
    img_a = load_image(path_a)
    img_b = load_image(path_b)

    img_a, img_b = resize_to_match(img_a, img_b, target_size)

    # Validate dimensions match
    if img_a.shape != img_b.shape:
        raise DimensionMismatchError(
            f"Dimension mismatch after resize: A={img_a.shape}, B={img_b.shape}"
        )

    h, w = img_a.shape

    return ImagePair(
        image_a=img_a,
        image_b=img_b,
        vector_a=flatten_image(img_a),
        vector_b=flatten_image(img_b),
        height=h,
        width=w,
        path_a=str(path_a),
        path_b=str(path_b),
    )
