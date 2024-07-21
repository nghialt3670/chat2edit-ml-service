from base64 import b64decode
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from utils.image import image_to_base64


class MaskModel(BaseModel):
    base64: str
    original_size: Optional[Tuple[int, int]] = Field(default=None)
    offset: Optional[Tuple[int, int]] = Field(default=None)

    @classmethod
    def from_array(cls, array: np.ndarray) -> "MaskModel":
        mask_image = Image.fromarray(array)
        mask_box = mask_image.getbbox()
        base64 = image_to_base64(mask_image.crop(mask_box))
        original_size = mask_image.size
        offset = mask_box[0], mask_box[1]
        return cls(base64=base64, original_size=original_size, offset=offset)

    def to_array(self) -> np.ndarray:
        buffer = b64decode(self.base64)
        array = np.frombuffer(buffer, np.uint8)
        array = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
        if self.original_size and self.offset:
            pad_width = (
                (
                    self.offset[1],
                    self.original_size[1] - array.shape[0] - self.offset[1],
                ),
                (
                    self.offset[0],
                    self.original_size[0] - array.shape[1] - self.offset[0],
                ),
            )
            array = np.pad(array, pad_width, mode="constant", constant_values=0)
        return array
