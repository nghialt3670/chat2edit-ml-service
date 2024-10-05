from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from config.predictors import PREDICTOR_INITS
from core.inference.manage.predictor_manager import PredictorManager
from core.inference.predictors.predictor import Predictor

manager = PredictorManager(PREDICTOR_INITS)


@asynccontextmanager
async def get_predictor(name: str) -> AsyncGenerator[Predictor, Any]:
    predictor = await manager.get(name)
    try:
        yield predictor
    finally:
        manager.claim(predictor)
