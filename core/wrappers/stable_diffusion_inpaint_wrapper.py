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
        original_size = image.size

        # Run the inpainting pipeline
        result_image = self.pipe(
            prompt=prompt,
            image=image.convert("RGB"),
            mask_image=PIL.Image.fromarray(mask),
        ).images[0]

        # Resize the result image back to the original size
        result_image = result_image.resize(original_size, PIL.Image.LANCZOS)

        return result_image
