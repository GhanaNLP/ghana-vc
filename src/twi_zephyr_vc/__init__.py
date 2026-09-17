"""Convert any Hugging Face audio dataset into the Twi *Zephyr* voice."""

from .config import (
    ALLOWED_DIFFUSION_STEPS,
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_MODEL_REPO,
)
from .dataset import convert_dataset, detect_audio_column
from .engine import ZephyrConverter, ensure_seedvc

__version__ = "0.1.0"

__all__ = [
    "ZephyrConverter",
    "convert_dataset",
    "detect_audio_column",
    "ensure_seedvc",
    "DEFAULT_DIFFUSION_STEPS",
    "ALLOWED_DIFFUSION_STEPS",
    "DEFAULT_MODEL_REPO",
    "__version__",
]
