from PIL import ImageFilter
from PIL.Image import Image
from simple_lama_inpainting import SimpleLama

from core.inference.predictors.predictor import Predictor


class LaMaPredictor(Predictor):
    def __init__(self, device: str) -> None:
        super().__init__(device)
        self._model = SimpleLama(device)

    def __call__(self, image: Image, mask: Image) -> Image:
        expanded_mask = self._expand_mask(mask)
        inpainted_image = self._model(image, expanded_mask)

        return inpainted_image

    def _expand_mask(self, image: Image) -> Image:
        if image.mode != "L":
            image = image.convert("L")

        blurred_mask = image.filter(ImageFilter.GaussianBlur(6))
        expanded_mask = blurred_mask.point(lambda p: 255 if p > 0 else 0)

        return expanded_mask
