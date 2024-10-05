from abc import ABC


class Predictor(ABC):
    def __init__(self, device: str) -> None:
        self._device = device
