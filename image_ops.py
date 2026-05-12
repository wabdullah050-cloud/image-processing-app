"""Image processing operations for the Streamlit image processing app."""

from __future__ import annotations

import cv2
import numpy as np


def _ensure_odd(kernel_size: int) -> int:
    """Return a valid odd kernel size accepted by OpenCV."""

    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def grayscale_conversion(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""

    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binary_conversion(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Convert a BGR image to a binary image using a threshold."""

    gray_image = grayscale_conversion(image)
    _, binary_image = cv2.threshold(gray_image, int(threshold), 255, cv2.THRESH_BINARY)
    return binary_image


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image to the requested width and height."""

    width = max(1, int(width))
    height = max(1, int(height))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur using an automatically corrected odd kernel size."""

    kernel_size = _ensure_odd(kernel_size)
    sigma = max(0.0, float(sigma))
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median blur using an automatically corrected odd kernel size."""

    kernel_size = _ensure_odd(kernel_size)
    return cv2.medianBlur(image, kernel_size)


def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
    """Return Sobel edge magnitude as a grayscale image."""

    gray_image = grayscale_conversion(image)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    if magnitude.max() > 0:
        magnitude = np.uint8(np.clip(magnitude / magnitude.max() * 255, 0, 255))
    else:
        magnitude = np.zeros_like(gray_image)
    return magnitude


def canny_edge_detection(image: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
    """Apply Canny edge detection to the image."""

    gray_image = grayscale_conversion(image)
    return cv2.Canny(gray_image, int(threshold1), int(threshold2))


def adjust_brightness_contrast(image: np.ndarray, brightness: int = 0, contrast: float = 1.0) -> np.ndarray:
    """Adjust brightness and contrast using OpenCV scaling."""

    return cv2.convertScaleAbs(image, alpha=float(contrast), beta=int(brightness))


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Equalize the histogram of a grayscale version of the image."""

    gray_image = grayscale_conversion(image)
    return cv2.equalizeHist(gray_image)