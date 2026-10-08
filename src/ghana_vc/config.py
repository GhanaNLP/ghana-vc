"""Defaults for ghana-vc."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

SEEDVC_REPO = "https://github.com/Plachtaa/seed-vc.git"
# Pinned to the commit this project is tested against. Seed-VC's
# requirements.txt drives the whole dependency stack (torch 2.4.0,
# numpy==1.26.4, an older huggingface_hub API used by BigVGAN), so tracking
# its main branch means an upstream change can silently break installs.
SEEDVC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"

DEFAULT_MODEL_REPO = "ghanaopenai/ghana-vc"

# Per-language models follow ghanaopenai/ghana-vc-<config>, one per language
# in ghanaopenai/ghana-synthetic-speech. Auto-selected when the caller names
# a language (config name); falls back to DEFAULT_MODEL_REPO (cross-lingual
# Twi checkpoint) when the per-language model has not been published.
MODEL_REPO_PREFIX = "ghanaopenai/ghana-vc-"

# Each per-language repo publishes a single ft_model.pth: the step-1000
# checkpoint, which kept pronunciation clearest (lowest CER) in a comparison of
# checkpoints from 500 to 3500 steps.

# 50 is the recommended setting. Listening tests on this checkpoint found 25
# (the Seed-VC default) audibly robotic, and 100 only marginally better than 50
# for roughly double the compute.
DEFAULT_DIFFUSION_STEPS = 50
ALLOWED_DIFFUSION_STEPS = (25, 50, 100)

DEFAULT_LENGTH_ADJUST = 1.0
DEFAULT_CFG_RATE = 0.7

# Column added to the output dataset.
OUTPUT_AUDIO_COLUMN = "audio_zephyr"

# Audio columns are auto-detected; these names are tried first.
COMMON_AUDIO_COLUMNS = ("audio", "audio_file", "speech", "wav", "sound")


def model_repo_for(config_name: str | None, token: str | None = None) -> str:
    """Resolve the best model repo for a language config name.

    Returns the per-language repo ``ghanaopenai/ghana-vc-<config>`` when that
    model exists on the Hub, otherwise the default cross-lingual model. Passing
    an explicit ``--model-repo`` at the CLI bypasses this entirely.
    """
    if not config_name:
        return DEFAULT_MODEL_REPO
    from huggingface_hub import HfApi

    candidate = f"{MODEL_REPO_PREFIX}{config_name}"
    try:
        HfApi(token=token).model_info(candidate)
    except Exception as exc:
        log.warning("No per-language model %s (%s); using cross-lingual %s",
                    candidate, type(exc).__name__, DEFAULT_MODEL_REPO)
        return DEFAULT_MODEL_REPO
    log.info("Using per-language model %s", candidate)
    return candidate
