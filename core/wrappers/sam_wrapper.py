from typing import List, Literal, Optional, Tuple

import numpy as np
from PIL.Image import Image
from segment_anything import SamPredictor, sam_model_registry


class SamWrapper:
    def __init__(
        self, model_type: Literal["vit_b", "vit_h"], checkpoint: str, device: str
    ) -> None:
        self.device = device
        sam = sam_model_registry[model_type](checkpoint)
        sam.to(device)
        self.model = SamPredictor(sam)

    def __call__(
        self,
        image: Image,
        box: Optional[Tuple[int, int, int, int]] = None,
        points: Optional[List[Tuple[int, int]]] = None,
        point_labels: Optional[List[int]] = None,
    ) -> np.ndarray:
        self.model.set_image(np.array(image.convert("RGB")))
        if points:
            points = np.array(points)
        if point_labels:
            point_labels = np.array(point_labels)
        if box:
            box = np.array(box)
        masks, _, _ = self.model.predict(
            points, point_labels, box, multimask_output=False
        )
        mask = masks[0].astype(np.uint8) * 255
        return mask
