from asyncio import Semaphore
from collections import deque
from typing import Any, Iterable


class ModelPool:
    def __init__(self, models: Iterable[Any]) -> None:
        self._models = models
        self._semaphore = Semaphore(len(models))
        self._queue = deque(models)

    async def get(self):
        await self._semaphore.acquire()
        return self._queue.pop()

    def claim(self, model: Any) -> None:
        self._queue.append(model)
        self._semaphore.release()
