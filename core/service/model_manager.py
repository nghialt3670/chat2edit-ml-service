from typing import Any, Dict

from core.service.factory import Factory
from core.service.model_pool import ModelPool


class ModelManager:
    def __init__(self, factory: Factory, name_to_quantity: Dict[str, int]) -> None:
        self._factory = factory
        self._name_to_quantity = name_to_quantity
        self._name_to_pool = {}
        self._id_to_name = {}

    def get_name_to_quantity(self) -> Dict[str, int]:
        return self._name_to_quantity

    def create_models(
        self,
    ) -> None:
        for name, quantity in self._name_to_quantity.items():
            models = []
            for _ in range(quantity):
                model = self._factory.create(name)
                self._id_to_name[id(model)] = name
                models.append(model)

            self._name_to_pool[name] = ModelPool(models)

    async def get(self, name: str) -> Any:
        return await self._name_to_pool[name].get()

    def claim(self, model: Any) -> None:
        class_name = self._id_to_name[id(model)]
        self._name_to_pool[class_name].claim(model)
