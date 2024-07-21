from typing import Dict, Optional, Union

from core.utils.helpers import read_yaml
from pydantic import BaseModel, Field


class ReferenceParam(BaseModel):
    class_name: str


class InitConfig(BaseModel):
    name: str
    class_path: str
    init_params: Optional[Dict[str, Union[int, float, str, ReferenceParam]]] = Field(
        default_factory=dict
    )

    def get_class_name(self) -> str:
        return self.class_path.split(".")[-1]

    def get_module_path(self) -> str:
        return self.class_path.rsplit(".", 1)[0]

    @classmethod
    def from_yaml(cls, file_path: str) -> "InitConfig":
        config_dict = read_yaml(file_path)
        return cls(**config_dict)
