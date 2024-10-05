from typing import Any, Dict, Type

from pydantic import BaseModel


class PredictorInit(BaseModel):
    type: Type
    params: Dict[str, Any]
