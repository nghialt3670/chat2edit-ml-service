from contextlib import asynccontextmanager
from typing import Any

from core.service.factory import Factory
from core.service.model_manager import ModelManager

MODEL_CONFIG_FILES = [
    "core/config/grounding_dino.yaml",
    "core/config/lama.yaml",
    "core/config/sam.yaml",
]

MODEL_NAME_TO_QUANTITY = {"grounding_dino": 1, "lama": 1, "sam": 1}

model_factory = Factory(MODEL_CONFIG_FILES)
model_manager = ModelManager(model_factory, MODEL_NAME_TO_QUANTITY)


async def get_model_manager() -> Any:
    return model_manager


@asynccontextmanager
async def get_model(model_manager: Any, name: str) -> Any:
    model = await model_manager.get(name)
    try:
        yield model
    finally:
        model_manager.claim(model)
