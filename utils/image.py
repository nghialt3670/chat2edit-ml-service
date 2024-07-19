from base64 import b64decode, b64encode
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def image_to_base64(image: Image.Image) -> str:
    image_bytes = BytesIO()
    image.format = image.format or "PNG"
    image.save(image_bytes, image.format)
    return b64encode(image_bytes.getvalue()).decode()


def base64_to_image(base64: str) -> Image.Image:
    image_bytes = BytesIO(b64decode(base64))
    return Image.open(image_bytes)


def expand_mask(mask: np.ndarray, iterations: int = 10) -> np.ndarray:
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated_mask
