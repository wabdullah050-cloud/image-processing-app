# Web-Based Digital Image Processing Application Using Python and OpenCV

A Streamlit web application for basic digital image processing. Users can upload an image, apply common spatial-domain operations, preview the results side by side, inspect histograms, and download the processed output.

## Features

- Upload JPG, JPEG, or PNG images
- Convert RGB images to grayscale
- Convert grayscale images to binary
- Apply Gaussian blur
- Apply median filtering
- Detect edges with Sobel and Canny
- Adjust brightness and contrast
- Display pixel intensity histograms
- Perform histogram equalization
- Resize images
- Download the processed image as PNG

## Tech Stack

- Python
- OpenCV
- Streamlit
- NumPy
- Pillow
- Matplotlib

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Verify imports:

```bash
python test_imports.py
```

4. Run the app:

```bash
streamlit run main.py
```

## Deployment Note

For Streamlit Cloud, this project uses `opencv-python-headless` in `requirements.txt` because the regular OpenCV package depends on GUI libraries that are not available on cloud servers.