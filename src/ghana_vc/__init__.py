"""Convert any Hugging Face audio dataset into the Twi *Zephyr* voice."""

from .config import (
    ALLOWED_DIFFUSION_STEPS,
    DEFAULT_CHECKPOINT_STEP,
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_MODEL_REPO,
    MODEL_REPO_PREFIX,
    model_repo_for,
)
from .dataset import convert_dataset, detect_audio_column
from .engine import ZephyrConverter, ensure_seedvc
from .local import collect_audio, convert_paths

__version__ = "0.1.0"

__all__ = [
    "ZephyrConverter",
    "convert_dataset",
    "detect_audio_column",
    "ensure_seedvc",
    "convert_paths",
    "collect_audio",
    "DEFAULT_DIFFUSION_STEPS",
    "ALLOWED_DIFFUSION_STEPS",
    "DEFAULT_CHECKPOINT_STEP",
    "DEFAULT_MODEL_REPO",
    "MODEL_REPO_PREFIX",
    "model_repo_for",
    "__version__",
]
