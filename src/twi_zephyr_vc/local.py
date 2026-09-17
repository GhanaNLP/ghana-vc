"""Convert local audio files or folders into the Zephyr voice.

The Hub path in :mod:`twi_zephyr_vc.dataset` is the convenient one when the
audio already lives on Hugging Face. This module covers the other common case:
a folder of wavs on disk, with the option to publish the result afterwards.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

from .config import DEFAULT_DIFFUSION_STEPS, DEFAULT_MODEL_REPO
from .engine import ZephyrConverter

log = logging.getLogger(__name__)

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma"}


def collect_audio(inputs: Sequence[str | Path], recursive: bool = True) -> list[Path]:
    """Expand files and directories into a sorted list of audio paths."""
    found: list[Path] = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            it: Iterable[Path] = p.rglob("*") if recursive else p.glob("*")
            found.extend(
                f for f in it if f.is_file() and f.suffix.lower() in AUDIO_SUFFIXES
            )
        elif p.is_file():
            if p.suffix.lower() not in AUDIO_SUFFIXES:
                log.warning("Skipping %s (unrecognised audio extension)", p)
            else:
                found.append(p)
        else:
            raise FileNotFoundError(f"No such file or directory: {p}")
    return sorted(set(found))


def convert_paths(
    inputs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    recursive: bool = True,
    diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
    model_repo: str = DEFAULT_MODEL_REPO,
    token: str | None = None,
    install_deps: bool = True,
    overwrite: bool = False,
    push_to: str | None = None,
    private: bool = False,
) -> list[Path]:
    """Convert local audio into the Zephyr voice.

    Directory structure under each input folder is preserved in ``output_dir``.
    Returns the list of files written.

    If ``push_to`` is given, the converted audio is also published to the Hub as
    a dataset with ``audio_zephyr`` plus the source filename.
    """
    import soundfile as sf

    files = collect_audio(inputs, recursive=recursive)
    if not files:
        raise ValueError(f"No audio files found in: {', '.join(str(i) for i in inputs)}")

    out_root = Path(output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    # Preserve structure relative to each input directory; bare files go flat.
    roots = [Path(i).expanduser() for i in inputs]
    dir_roots = [r for r in roots if r.is_dir()]

    def dest_for(src: Path) -> Path:
        for root in dir_roots:
            try:
                return out_root / src.relative_to(root).with_suffix(".wav")
            except ValueError:
                continue
        return out_root / (src.stem + ".wav")

    converter = ZephyrConverter(
        diffusion_steps=diffusion_steps,
        model_repo=model_repo,
        token=token,
        install_deps=install_deps,
    )
    converter.load()

    written: list[Path] = []
    failures: list[Path] = []

    log.info("Converting %d file(s) at %d diffusion steps", len(files), diffusion_steps)
    for i, src in enumerate(files, 1):
        dst = dest_for(src)
        if dst.exists() and not overwrite:
            log.info("[%d/%d] skip (exists): %s", i, len(files), dst)
            written.append(dst)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            wav, sr = converter.convert_file(src)
            sf.write(dst, wav, sr)
            written.append(dst)
            log.info("[%d/%d] %s -> %s", i, len(files), src.name, dst)
        except Exception as exc:  # one bad file shouldn't end a long run
            failures.append(src)
            log.warning("[%d/%d] FAILED %s: %s", i, len(files), src.name, exc)

    if failures:
        log.warning("%d/%d file(s) failed: %s",
                    len(failures), len(files), [f.name for f in failures[:20]])

    if push_to:
        _push(written, files, push_to, diffusion_steps, model_repo, token, private)

    return written


def _push(written, sources, repo_id, diffusion_steps, model_repo, token, private):
    """Publish converted local audio to the Hub as a dataset."""
    from datasets import Audio, Dataset

    log.info("Pushing %d converted file(s) to %s", len(written), repo_id)
    by_stem = {s.stem: s for s in sources}
    ds = Dataset.from_dict(
        {
            "audio_zephyr": [str(p) for p in written],
            "source_file": [by_stem.get(p.stem, p).name for p in written],
            "zephyr_diffusion_steps": [diffusion_steps] * len(written),
            "zephyr_model": [model_repo] * len(written),
        }
    ).cast_column("audio_zephyr", Audio())
    ds.push_to_hub(repo_id, token=token, private=private)
    log.info("Done: https://huggingface.co/datasets/%s", repo_id)
