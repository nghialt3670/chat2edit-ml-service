from asyncio import Semaphore
from collections import deque
from typing import Any, Iterable

from core.inference.predictors.predictor import Predictor


class PredictorPool:
    def __init__(self, predictors: Iterable[Predictor]) -> None:
        self._semaphore = Semaphore(len(predictors))
        self._queue = deque(predictors)

    async def get(self):
        await self._semaphore.acquire()
        return self._queue.pop()

    def add(self, predictor: Predictor) -> None:
        self._queue.append(predictor)
        curr_len = len(self._queue)
        self._semaphore = Semaphore(curr_len)

    def claim(self, model: Any) -> None:
        self._queue.append(model)
        self._semaphore.release()
