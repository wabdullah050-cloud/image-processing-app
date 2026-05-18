"""Streamlit web app for basic digital image processing."""

from __future__ import annotations

import io

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

OPERATIONS = [
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
]


def _load_uploaded_image(uploaded_file) -> tuple[np.ndarray, int]:
    """Read an uploaded file into a BGR numpy array and return the file size."""

    file_size_bytes = uploaded_file.size or 0
    pil_image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    original_rgb = np.array(pil_image)
    return cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR), file_size_bytes


def _load_uploaded_bytes(image_bytes: bytes) -> np.ndarray:
    """Read cached image bytes into a BGR numpy array."""

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_rgb = np.array(pil_image)
    return cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)


def process_current_image(image: np.ndarray, operation: str) -> np.ndarray:
    """Run the selected operation and return the processed image."""

    if operation == "Original":
        return image

    if operation == "Grayscale Conversion":
        return grayscale_conversion(image)

    if operation == "Binary Conversion":
        threshold = st.sidebar.slider("Threshold", 0, 255, 127, key="threshold")
        return binary_conversion(image, threshold=threshold)

    if operation == "Gaussian Blur":
        kernel_size = st.sidebar.slider("Kernel size", 1, 31, 5, step=2, key="gaussian_kernel")
        sigma = st.sidebar.slider("Sigma", 0.1, 10.0, 1.0, 0.1, key="gaussian_sigma")
        return apply_gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)

    if operation == "Median Filter":
        kernel_size = st.sidebar.slider("Kernel size", 1, 31, 5, step=2, key="median_kernel")
        return apply_median_blur(image, kernel_size=kernel_size)

    if operation == "Sobel Edge Detection":
        return sobel_edge_detection(image)

    if operation == "Canny Edge Detection":
        threshold1 = st.sidebar.slider("Threshold 1", 0, 255, 100, key="canny_threshold1")
        threshold2 = st.sidebar.slider("Threshold 2", 0, 255, 200, key="canny_threshold2")
        return canny_edge_detection(image, threshold1=threshold1, threshold2=threshold2)

    if operation == "Brightness & Contrast":
        brightness = st.sidebar.slider("Brightness", -100, 100, 0, key="brightness")
        contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0, 0.1, key="contrast")
        return adjust_brightness_contrast(image, brightness=brightness, contrast=contrast)

    if operation == "Histogram Equalization":
        return histogram_equalization(image)

    if operation == "Resize":
        image_height, image_width = image.shape[:2]
        width = st.sidebar.number_input("Width", min_value=1, value=int(image_width), key="resize_width")
        height = st.sidebar.number_input("Height", min_value=1, value=int(image_height), key="resize_height")
        return resize_image(image, width=int(width), height=int(height))

    return image


def _operation_controls(operation: str, image: np.ndarray) -> dict:
    """Render sidebar controls for a single operation and return its parameters."""

    if operation == "Binary Conversion":
        return {"threshold": st.sidebar.slider("Threshold", 0, 255, 127, key="threshold")}

    if operation == "Gaussian Blur":
        return {
            "kernel_size": st.sidebar.slider("Kernel size", 1, 31, 5, step=2, key="gaussian_kernel"),
            "sigma": st.sidebar.slider("Sigma", 0.1, 10.0, 1.0, 0.1, key="gaussian_sigma"),
        }

    if operation == "Median Filter":
        return {"kernel_size": st.sidebar.slider("Kernel size", 1, 31, 5, step=2, key="median_kernel")}

    if operation == "Canny Edge Detection":
        return {
            "threshold1": st.sidebar.slider("Threshold 1", 0, 255, 100, key="canny_threshold1"),
            "threshold2": st.sidebar.slider("Threshold 2", 0, 255, 200, key="canny_threshold2"),
        }

    if operation == "Brightness & Contrast":
        return {
            "brightness": st.sidebar.slider("Brightness", -100, 100, 0, key="brightness"),
            "contrast": st.sidebar.slider("Contrast", 0.5, 3.0, 1.0, 0.1, key="contrast"),
        }

    if operation == "Resize":
        image_height, image_width = image.shape[:2]
        return {
            "width": st.sidebar.number_input("Width", min_value=1, value=int(image_width), key="resize_width"),
            "height": st.sidebar.number_input("Height", min_value=1, value=int(image_height), key="resize_height"),
        }

    return {}


def _apply_operation(image: np.ndarray, operation: str, params: dict) -> np.ndarray:
    """Apply one selected operation using the provided parameters."""

    if operation == "Original":
        return image

    if operation == "Grayscale Conversion":
        return grayscale_conversion(image)

    if operation == "Binary Conversion":
        return binary_conversion(image, threshold=int(params.get("threshold", 127)))

    if operation == "Gaussian Blur":
        return apply_gaussian_blur(
            image,
            kernel_size=int(params.get("kernel_size", 5)),
            sigma=float(params.get("sigma", 1.0)),
        )

    if operation == "Median Filter":
        return apply_median_blur(image, kernel_size=int(params.get("kernel_size", 5)))

    if operation == "Sobel Edge Detection":
        return sobel_edge_detection(image)

    if operation == "Canny Edge Detection":
        return canny_edge_detection(
            image,
            threshold1=int(params.get("threshold1", 100)),
            threshold2=int(params.get("threshold2", 200)),
        )

    if operation == "Brightness & Contrast":
        return adjust_brightness_contrast(
            image,
            brightness=int(params.get("brightness", 0)),
            contrast=float(params.get("contrast", 1.0)),
        )

    if operation == "Histogram Equalization":
        return histogram_equalization(image)

    if operation == "Resize":
        return resize_image(
            image,
            width=int(params.get("width", image.shape[1])),
            height=int(params.get("height", image.shape[0])),
        )

    return image


def run_app() -> None:
    """Render the Streamlit app."""

    st.set_page_config(page_title="Digital Image Processing App", page_icon="🖼️", layout="wide")

    st.title("Web-Based Digital Image Processing Application")
    st.caption("Python + OpenCV + Streamlit")

    st.markdown(
        """
        Use the sidebar to upload an image and choose one or more processing techniques.
        Each selected technique is rendered below in its own section so you can scroll,
        compare, and deselect techniques without losing the uploaded image.
        """
    )

    selected_operations = st.sidebar.multiselect(
        "Choose processing techniques",
        [operation for operation in OPERATIONS if operation != "Original"],
        default=["Grayscale Conversion"],
        key="operations",
    )

    uploaded_file = st.sidebar.file_uploader("Upload a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"], key="uploaded_image")

    cached_bytes = st.session_state.get("cached_uploaded_bytes")
    cached_size = st.session_state.get("cached_uploaded_size")

    if uploaded_file is not None:
        try:
            original_image, file_size_bytes = _load_uploaded_image(uploaded_file)
            st.session_state["cached_uploaded_bytes"] = uploaded_file.getvalue()
            st.session_state["cached_uploaded_size"] = file_size_bytes
        except UnidentifiedImageError:
            st.error("The uploaded file is not a valid image.")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unable to load the image: {exc}")
            return
    elif cached_bytes is not None:
        try:
            file_size_bytes = cached_size or len(cached_bytes)
            original_image = _load_uploaded_bytes(cached_bytes)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unable to reload the cached image: {exc}")
            return
    else:
        st.info("Upload an image from the sidebar to begin.")
        return

    if file_size_bytes > 10 * 1024 * 1024:
        st.warning("The uploaded image is larger than 10 MB. Processing may be slower.")

    operation_params: dict[str, dict] = {}
    st.sidebar.markdown("---")
    if selected_operations:
        st.sidebar.write("Select or deselect techniques to compare multiple results.")
        for operation in selected_operations:
            with st.sidebar.expander(operation, expanded=operation in {"Grayscale Conversion", "Binary Conversion"}):
                st.caption(OPERATION_DESCRIPTIONS[operation])
                operation_params[operation] = _operation_controls(operation, original_image)
    else:
        st.sidebar.write("Select one or more techniques to generate results below.")

    if selected_operations:
        st.info("Selected techniques will appear one below another as you scroll down.")
    else:
        st.info("Choose one or more image processing techniques from the sidebar.")

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Original Image")
        st.image(image_to_display(original_image), width="stretch")
        st.caption(f"Dimensions: {get_image_dimensions(original_image)}")
        st.caption(f"File size: {file_size_bytes / (1024 * 1024):.2f} MB")

    with right_column:
        if not selected_operations:
            st.subheader("Processed Image")
            st.info("No processing technique selected yet.")
            return

        for operation in selected_operations:
            st.markdown("---")
            st.subheader(operation)
            st.caption(OPERATION_DESCRIPTIONS[operation])

            try:
                processed_image = _apply_operation(original_image, operation, operation_params.get(operation, {}))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Processing failed for {operation}: {exc}")
                continue

            st.image(image_to_display(processed_image), width="stretch")
            st.caption(f"Dimensions: {get_image_dimensions(processed_image)}")

            if operation == "Resize":
                st.caption("The resize operation returns the selected output dimensions.")

            histogram_figure = plot_histogram(processed_image)
            st.pyplot(histogram_figure, width="stretch")

            processed_bytes = image_to_png_bytes(processed_image)
            st.download_button(
                label=f"Download {operation.lower()} image",
                data=processed_bytes,
                file_name=f"processed_{operation.lower().replace(' & ', '_').replace(' ', '_')}.png",
                mime="image/png",
                key=f"download_{operation}",
            )


if __name__ == "__main__":
    run_app()