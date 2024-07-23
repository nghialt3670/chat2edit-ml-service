import numpy as np
import PIL
import PIL.Image
import torch
from diffusers import StableDiffusionInpaintPipeline


class StableDiffusionInpaintWrapper:
    def __init__(self, hf_repo_id: str, device: str) -> None:
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            hf_repo_id, torch_dtype=torch.float32
        ).to(device)

    def __call__(
        self, image: PIL.Image.Image, mask: np.ndarray, prompt: str
    ) -> PIL.Image.Image:
        return self.pipe(
            prompt=prompt,
            image=image.convert("RGB"),
            mask_image=PIL.Image.fromarray(mask),
        ).images[0]
