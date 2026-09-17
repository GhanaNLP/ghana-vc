"""Defaults for twi-zephyr-vc."""

from __future__ import annotations

SEEDVC_REPO = "https://github.com/Plachtaa/seed-vc.git"
SEEDVC_COMMIT = ""  # pin a commit here if upstream drift becomes a problem

DEFAULT_MODEL_REPO = "ghanaopenai/twi-zephyr-vc"

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
