"""Defaults for ghana-vc."""

from __future__ import annotations

SEEDVC_REPO = "https://github.com/Plachtaa/seed-vc.git"
# Pinned to the commit this project is tested against. Seed-VC's
# requirements.txt drives the whole dependency stack (torch 2.4.0,
# numpy==1.26.4, an older huggingface_hub API used by BigVGAN), so tracking
# its main branch means an upstream change can silently break installs.
SEEDVC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"

DEFAULT_MODEL_REPO = "ghanaopenai/ghana-vc"

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
