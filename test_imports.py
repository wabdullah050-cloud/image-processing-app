"""Quick import verification script for the project environment."""

import cv2
import matplotlib
import numpy as np
import PIL
import streamlit


def main() -> None:
    print(f"cv2={cv2.__version__}")
    print(f"numpy={np.__version__}")
    print(f"Pillow={PIL.__version__}")
    print(f"matplotlib={matplotlib.__version__}")
    print(f"streamlit={streamlit.__version__}")
    print("OK")


if __name__ == "__main__":
    main()