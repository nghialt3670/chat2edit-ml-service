import PIL.Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL.Image import Image

from core.inference.predictors.predictor import Predictor


class SDInpaintPredictor(Predictor):
    def __init__(self, device: str, repo_id: str) -> None:
        super().__init__(device)
        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            repo_id, torch_dtype=torch.float32
        )
        self._pipe.to(device)

    def __call__(self, image: Image, mask: Image, prompt: str) -> Image:
        image = image.convert("RGB")

        result_image = self._pipe(
            image=image,
            mask_image=mask,
            prompt=prompt,
        ).images[0]

        resized_image = result_image.resize(image.size, PIL.Image.LANCZOS)
        
        return resized_image
