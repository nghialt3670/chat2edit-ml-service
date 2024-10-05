from typing import List, Optional

import numpy as np
import PIL.Image
import PIL.ImageFilter
from PIL.Image import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sympy import Point

from core.inference.predictors.predictor import Predictor
from core.types.box import Box


class SAM2Predictor(Predictor):
    def __init__(self, device: str, config: str, checkpoint: str) -> None:
        super().__init__(device)
        model = build_sam2(config, checkpoint, device=device)
        self._predictor = SAM2ImagePredictor(model)

    def __call__(
        self,
        image: Image,
        box: Optional[Box] = None,
        points: Optional[List[Point]] = None,
        point_labels: Optional[List[int]] = None,
    ) -> Image:
        image = image.convert("RGB")
        image_array = np.array(image)

        self._predictor.set_image(image_array)

        if box:
            box = np.array(box)
        if points:
            points = np.array(points)
        if point_labels:
            point_labels = np.array(point_labels)

        masks, _, _ = self._predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            box=box,
            multimask_output=False,
        )

        mask_array = masks[0].astype(np.uint8) * 255
        mask = PIL.Image.fromarray(mask_array)
        processed_mask = self._post_process_mask(mask)

        return processed_mask

    def _post_process_mask(self, image: Image) -> Image:
        if image.mode != "L":
            image = image.convert("L")

        blurred_mask = image.filter(PIL.ImageFilter.GaussianBlur(3))
        thesholded_mask = blurred_mask.point(lambda p: 255 if p > 128 else 0)

        return thesholded_mask
