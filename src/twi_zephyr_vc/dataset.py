"""Convert a Hugging Face audio dataset into the Zephyr voice and push it back."""

from __future__ import annotations

import logging
from typing import Iterable

from .config import (
    COMMON_AUDIO_COLUMNS,
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_MODEL_REPO,
    OUTPUT_AUDIO_COLUMN,
)
from .engine import ZephyrConverter

log = logging.getLogger(__name__)


def detect_audio_column(dataset) -> str:
    """Find the audio column, preferring conventional names."""
    from datasets import Audio

    audio_cols = [
        name for name, feat in dataset.features.items() if isinstance(feat, Audio)
    ]
    if not audio_cols:
        raise ValueError(
            "No Audio column found in this dataset. Pass --audio-column "
            f"explicitly. Available columns: {list(dataset.features)}"
        )
    for preferred in COMMON_AUDIO_COLUMNS:
        if preferred in audio_cols:
            return preferred
    return audio_cols[0]


def convert_dataset(
    dataset_id: str,
    output_id: str,
    *,
    split: str = "train",
    config_name: str | None = None,
    audio_column: str | None = None,
    num_samples: int | None = None,
    diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
    model_repo: str = DEFAULT_MODEL_REPO,
    token: str | None = None,
    private: bool = False,
    keep_original: bool = True,
    install_deps: bool = True,
):
    """Convert ``dataset_id`` into the Zephyr voice and push to ``output_id``.

    The output keeps every original column and adds:

    ``audio_zephyr``
        the converted audio
    ``zephyr_diffusion_steps`` / ``zephyr_model``
        provenance, so a dataset can be traced back to how it was made
    """
    from datasets import Audio, load_dataset

    log.info("Loading %s (split=%s)", dataset_id, split)
    ds = load_dataset(dataset_id, config_name, split=split, token=token)

    if num_samples is not None and num_samples < len(ds):
        ds = ds.select(range(num_samples))
        log.info("Limited to %d samples", num_samples)

    column = audio_column or detect_audio_column(ds)
    log.info("Using audio column %r", column)

    converter = ZephyrConverter(
        diffusion_steps=diffusion_steps,
        model_repo=model_repo,
        token=token,
        install_deps=install_deps,
    )
    converter.load()  # load once, before the map

    failures: list[int] = []

    def _convert(example, idx):
        item = example[column]
        try:
            wav, sr = converter.convert_array(item["array"], item["sampling_rate"])
            example[OUTPUT_AUDIO_COLUMN] = {"array": wav, "sampling_rate": sr}
        except Exception as exc:  # keep going; one bad clip shouldn't end the run
            log.warning("Sample %d failed: %s", idx, exc)
            failures.append(idx)
            example[OUTPUT_AUDIO_COLUMN] = {
                "array": item["array"],
                "sampling_rate": item["sampling_rate"],
            }
        example["zephyr_diffusion_steps"] = diffusion_steps
        example["zephyr_model"] = model_repo
        return example

    log.info("Converting %d samples at %d diffusion steps", len(ds), diffusion_steps)
    out = ds.map(_convert, with_indices=True, desc="Converting to Zephyr")
    out = out.cast_column(OUTPUT_AUDIO_COLUMN, Audio())

    if not keep_original:
        out = out.remove_columns([column])

    if failures:
        log.warning(
            "%d/%d samples failed and kept their original audio: %s",
            len(failures), len(ds), failures[:20],
        )

    log.info("Pushing to %s", output_id)
    out.push_to_hub(output_id, token=token, private=private)
    log.info("Done: https://huggingface.co/datasets/%s", output_id)
    return out
