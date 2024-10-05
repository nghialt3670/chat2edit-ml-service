import logging
import os
from typing import Any, Dict, Iterable

from core.inference.manage.predictor_init import PredictorInit
from core.inference.manage.predictor_pool import PredictorPool
from core.inference.predictors.predictor import Predictor

logger = logging.getLogger("uvicorn")

class PredictorManager:
    def __init__(self, inits: Iterable[PredictorInit]) -> None:
        self._inits = inits
        self._name_to_pool: Dict[str, PredictorPool] = {}

    def init(self) -> None:
        for init in self._inits:
            name = init.type.__name__
            logger.info(f"Initializing {name}")
            predictor = init.type(**init.params)
            if not name in self._name_to_pool:
                self._name_to_pool[name] = PredictorPool([predictor])
            else:
                self._name_to_pool[name].add(predictor)

    async def get(self, name: str) -> Any:
        return await self._name_to_pool[name].get()

    def claim(self, predictor: Predictor) -> None:
        name = type(predictor).__name__
        self._name_to_pool[name].claim(predictor)
