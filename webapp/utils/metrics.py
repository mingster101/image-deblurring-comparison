"""Blur quality metrics used for evaluation."""
from __future__ import annotations
import numpy as np
from PIL import Image


def laplacian_variance(image: Image.Image) -> float:
    """
    Estimate image sharpness via Laplacian variance (edge-based).
    Higher value = sharper image (less blur).
    """
    import cv2
    img_np = np.array(image.convert('L'))  # grayscale
    lap = cv2.Laplacian(img_np, cv2.CV_64F)
    return float(lap.var())
