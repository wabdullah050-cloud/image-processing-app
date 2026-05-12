"""Streamlit web app for basic digital image processing."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from image_ops import (
    adjust_brightness_contrast,
    apply_gaussian_blur,
    apply_median_blur,
    binary_conversion,
    canny_edge_detection,
    grayscale_conversion,
    histogram_equalization,
    resize_image,
    sobel_edge_detection,
)
from utils import get_image_dimensions, image_to_display, image_to_png_bytes, plot_histogram


st.set_page_config(page_title="Digital Image Processing App", page_icon="🖼️", layout="wide")

st.title("Web-Based Digital Image Processing Application")
st.caption("Python + OpenCV + Streamlit")

st.markdown(
    """
    Use the sidebar to upload an image and choose a processing operation.
    The app supports grayscale, binary, filtering, edge detection, brightness/contrast,
    histogram equalization, resizing, histogram visualization, and download.
    """
)

OPERATION_DESCRIPTIONS = {
    "Grayscale Conversion": "Converts the uploaded color image into a grayscale image.",
    "Binary Conversion": "Turns the image into black-and-white pixels using a threshold.",
    "Gaussian Blur": "Smooths the image to reduce noise with a Gaussian filter.",
    "Median Filter": "Removes salt-and-pepper noise while preserving edges.",
    "Sobel Edge Detection": "Finds edges by measuring horizontal and vertical intensity changes.",
    "Canny Edge Detection": "Detects strong edges using gradient-based thresholding.",
    "Brightness & Contrast": "Adjusts brightness and contrast in real time.",
    "Histogram Equalization": "Improves contrast by redistributing pixel intensities.",
    "Resize": "Changes the output dimensions to a custom width and height.",
    "Original": "Displays the uploaded image without processing.",
}

operation = st.sidebar.selectbox(
    "Choose an operation",
    [
        "Original",
        "Grayscale Conversion",
        "Binary Conversion",
        "Gaussian Blur",
        "Median Filter",
        "Sobel Edge Detection",
        "Canny Edge Detection",
        "Brightness & Contrast",
        "Histogram Equalization",
        "Resize",
    ],
)

st.sidebar.markdown("---")
st.sidebar.write(OPERATION_DESCRIPTIONS[operation])

uploaded_file = st.sidebar.file_uploader("Upload a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])

original_image = None
processed_image = None

if uploaded_file is None:
    st.info("Upload an image from the sidebar to begin.")
    st.stop()

try:
    file_size_bytes = uploaded_file.size or 0
    if file_size_bytes > 10 * 1024 * 1024:
        st.warning("The uploaded image is larger than 10 MB. Processing may be slower.")

    pil_image = Image.open(uploaded_file).convert("RGB")
    original_rgb = np.array(pil_image)
    original_image = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
except UnidentifiedImageError:
    st.error("The uploaded file is not a valid image.")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Unable to load the image: {exc}")
    st.stop()

st.info(OPERATION_DESCRIPTIONS[operation])

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Original Image")
    st.image(image_to_display(original_image), use_container_width=True)
    st.caption(f"Dimensions: {get_image_dimensions(original_image)}")
    st.caption(f"File size: {file_size_bytes / (1024 * 1024):.2f} MB")


def process_current_image(image: np.ndarray) -> np.ndarray:
    """Run the selected operation and return the processed image."""

    if operation == "Original":
        return image

    if operation == "Grayscale Conversion":
        return grayscale_conversion(image)

    if operation == "Binary Conversion":
        threshold = st.sidebar.slider("Threshold", 0, 255, 127)
        return binary_conversion(image, threshold=threshold)

    if operation == "Gaussian Blur":
        kernel_size = st.sidebar.slider("Kernel size", 1, 31, 5, step=2)
        sigma = st.sidebar.slider("Sigma", 0.1, 10.0, 1.0, 0.1)
        return apply_gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)

    if operation == "Median Filter":
        kernel_size = st.sidebar.slider("Kernel size", 1, 31, 5, step=2)
        return apply_median_blur(image, kernel_size=kernel_size)

    if operation == "Sobel Edge Detection":
        return sobel_edge_detection(image)

    if operation == "Canny Edge Detection":
        threshold1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
        threshold2 = st.sidebar.slider("Threshold 2", 0, 255, 200)
        return canny_edge_detection(image, threshold1=threshold1, threshold2=threshold2)

    if operation == "Brightness & Contrast":
        brightness = st.sidebar.slider("Brightness", -100, 100, 0)
        contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0, 0.1)
        return adjust_brightness_contrast(image, brightness=brightness, contrast=contrast)

    if operation == "Histogram Equalization":
        return histogram_equalization(image)

    if operation == "Resize":
        image_height, image_width = image.shape[:2]
        width = st.sidebar.number_input("Width", min_value=1, value=int(image_width))
        height = st.sidebar.number_input("Height", min_value=1, value=int(image_height))
        return resize_image(image, width=int(width), height=int(height))

    return image


try:
    processed_image = process_current_image(original_image)
except Exception as exc:  # noqa: BLE001
    st.error(f"Processing failed: {exc}")
    st.stop()

with right_column:
    st.subheader("Processed Image")
    st.image(image_to_display(processed_image), use_container_width=True)
    st.caption(f"Dimensions: {get_image_dimensions(processed_image)}")

    if operation == "Resize":
        st.caption("The resize operation returns the selected output dimensions.")

    histogram_figure = plot_histogram(processed_image)
    st.pyplot(histogram_figure, use_container_width=True)

    processed_bytes = image_to_png_bytes(processed_image)
    st.download_button(
        label="Download processed image",
        data=processed_bytes,
        file_name=f"processed_{operation.lower().replace(' & ', '_').replace(' ', '_')}.png",
        mime="image/png",
    )