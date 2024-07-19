from typing import List, Tuple

import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.util.inference import load_model, predict
from PIL.Image import Image
from torchvision.ops import box_convert

from core.utils.helpers import suppress_stdout

TRANSFORM = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class GroundingDINOWrapper:
    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str,
        box_threshold: float,
        text_threshold: float,
    ) -> None:
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        with suppress_stdout():
            self.model = load_model(config, checkpoint, device)

    def __call__(
        self, image: Image, prompt: str
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        h, w, _ = np.asarray(image).shape
        caption = prompt + " ."
        boxes, logits, _ = predict(
            model=self.model,
            image=TRANSFORM(image.convert("RGB"), None)[0],
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        boxes = box_convert(
            boxes=boxes * torch.Tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy"
        )
        scores = list(map(float, logits))
        boxes = list(map(tuple, boxes.numpy().astype(int)))
        return boxes, scores
