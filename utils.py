"""Utility helpers for display, export, and plotting."""

from __future__ import annotations

import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def image_to_display(image: np.ndarray) -> np.ndarray:
    """Convert a cv2 BGR or grayscale image into a display-ready RGB/gray array."""

    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Convert a cv2 image into PNG bytes for download."""

    if image.ndim == 2:
        pil_image = Image.fromarray(image)
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def plot_histogram(image: np.ndarray):
    """Return a matplotlib figure showing the grayscale intensity histogram."""

    if image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    figure, axis = plt.subplots(figsize=(6, 3))
    axis.hist(gray_image.ravel(), bins=256, range=(0, 256), color="#1f77b4")
    axis.set_title("Pixel Intensity Histogram")
    axis.set_xlabel("Intensity")
    axis.set_ylabel("Frequency")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def get_image_dimensions(image: np.ndarray) -> str:
    """Return a human readable dimension string."""

    height, width = image.shape[:2]
    return f"{width} x {height}"