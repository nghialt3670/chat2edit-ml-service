from typing import List, Tuple

import torch
from PIL.Image import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from core.inference.predictors.predictor import Predictor
from core.types.box import Box


class GDinoPredictor(Predictor):
    def __init__(
        self, device: str, repo_id, box_threshold: float, text_threshold: float
    ) -> None:
        super().__init__(device)
        self._processor = AutoProcessor.from_pretrained(repo_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(repo_id)
        self._model.to(device)
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold

    def __call__(self, image: Image, prompt: str) -> Tuple[List[float], List[Box]]:
        text = prompt + " ."
        inputs = self._processor(images=image, text=text, return_tensors="pt")
        inputs.to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[image.size[::-1]],
        )

        scores = results[0]["scores"].tolist()
        boxes = [tuple(map(int, box)) for box in results[0]["boxes"].tolist()]

        return scores, boxes
