import cv2
import numpy as np
import PIL.Image
import torch
from iopaint.model import LaMa
from iopaint.schemas import InpaintRequest


class LaMaWrapper(LaMa):
    def __init__(self, checkpoint: str, device: str) -> None:
        self.model = torch.jit.load(checkpoint, "cpu").eval().to(device)
        self.device = device

    def __call__(self, image: PIL.Image.Image, mask: np.ndarray) -> PIL.Image.Image:
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        config = InpaintRequest(hd_strategy="Resize")
        inpainted_image = super().__call__(image, mask, config)
        return PIL.Image.fromarray(inpainted_image.astype(np.uint8))
