import os
import sys
from contextlib import contextmanager
from typing import Dict, Tuple

import numpy as np
import PIL
import PIL.Image
import yaml


@contextmanager
def suppress_stdout():
    save_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = save_stdout


def read_yaml(file_path: str) -> Dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)
