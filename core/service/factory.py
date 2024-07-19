from typing import Any, Iterable

from core.service.init_config import InitConfig, ReferenceParam


class Factory:
    def __init__(self, config_file_paths: Iterable[str]) -> None:
        configs = [InitConfig.from_yaml(file_path) for file_path in config_file_paths]
        self._name_to_config = {config.name: config for config in configs}

    def create(self, name: str) -> Any:
        config = self._name_to_config[name]
        module_path = config.get_module_path()
        class_name = config.get_class_name()
        module = __import__(module_path, fromlist=[class_name])
        class_obj = getattr(module, class_name)
        init_params = config.init_params.copy()
        for key, value in init_params.items():
            if isinstance(value, ReferenceParam):
                init_params[key] = self.create(value.class_name)
        instance = class_obj(**init_params)
        return instance

    def get_config(self, name: str) -> InitConfig:
        return self._name_to_config[name]
