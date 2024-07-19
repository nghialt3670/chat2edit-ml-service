import logging
import os
import sys
import warnings

sys.path.append(os.getcwd())
logger = logging.getLogger("uvicorn")


def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    logger.warning(f"{category.__name__}: {message} (from {filename}:{lineno})")


warnings.showwarning = custom_warning_handler
warnings.filterwarnings("always")
