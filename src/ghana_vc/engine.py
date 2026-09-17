"""Seed-VC engine wrapper.

Seed-VC ships as a git repository with an ``inference.py`` script rather than as
an installable package, and that script loads the checkpoint on every call. For
dataset-scale work we need the opposite: load once, convert many.

The approach here is to call the upstream ``load_models()`` a single time, cache
the result, and monkeypatch it so the upstream ``main()`` reuses the loaded
models. That keeps the actual conversion maths as upstream wrote it -- chunking,
overlap crossfade and all -- while removing the per-clip reload.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

from .config import (
    DEFAULT_CFG_RATE,
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_LENGTH_ADJUST,
    DEFAULT_MODEL_REPO,
    SEEDVC_COMMIT,
    SEEDVC_REPO,
)

log = logging.getLogger(__name__)


def _cache_root() -> Path:
    root = os.environ.get("GHANA_VC_HOME")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "ghana-vc"


def ensure_seedvc(install_deps: bool = True) -> Path:
    """Clone Seed-VC into the cache directory and return its path."""
    dest = _cache_root() / "seed-vc"
    if not (dest / "inference.py").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("Cloning Seed-VC into %s", dest)
        subprocess.run(["git", "clone", SEEDVC_REPO, str(dest)], check=True)
        if SEEDVC_COMMIT:
            subprocess.run(
                ["git", "-C", str(dest), "checkout", SEEDVC_COMMIT], check=True
            )
        if install_deps:
            log.info("Installing Seed-VC requirements")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r",
                 str(dest / "requirements.txt")],
                check=True,
            )
            _repair_hub_stack()
    return dest


# Seed-VC's requirements.txt pins an older huggingface_hub, which leaves
# `datasets` importing symbols that version doesn't have -- in practice
# "cannot import name 'XetDownloadProgressReporter'". Seed-VC itself only
# needs hf_hub_download, so restoring a current Hub stack afterwards is safe
# and keeps the dataset path working.
HUB_STACK = ["huggingface_hub>=0.34", "datasets>=2.18"]


def _repair_hub_stack() -> None:
    log.info("Restoring huggingface_hub / datasets after Seed-VC pins")
    # --upgrade-strategy only-if-needed is load-bearing: without it pip drags
    # numpy to 2.x, which breaks Seed-VC's numpy 1.x stack with
    # "ModuleNotFoundError: No module named 'numpy.strings'".
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
         "--upgrade-strategy", "only-if-needed", *HUB_STACK],
        check=False,
    )


class ZephyrConverter:
    """Converts speech into the Twi *Zephyr* voice.

    Parameters
    ----------
    diffusion_steps:
        Higher is smoother but slower. 50 is the recommended default; 25 is
        audibly robotic on this checkpoint and 100 gives only a small further
        gain for roughly double the time.
    """

    def __init__(
        self,
        diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
        model_repo: str = DEFAULT_MODEL_REPO,
        length_adjust: float = DEFAULT_LENGTH_ADJUST,
        cfg_rate: float = DEFAULT_CFG_RATE,
        fp16: bool = True,
        token: str | None = None,
        install_deps: bool = True,
    ) -> None:
        self.diffusion_steps = int(diffusion_steps)
        self.length_adjust = float(length_adjust)
        self.cfg_rate = float(cfg_rate)
        self.fp16 = bool(fp16)
        self.model_repo = model_repo
        self._token = token

        self._seed_dir = ensure_seedvc(install_deps=install_deps)
        self._inference = None
        self._reference_path: Path | None = None
        self._loaded = False

    # -- model loading -----------------------------------------------------
    def _download_assets(self) -> tuple[str, str, str]:
        from huggingface_hub import hf_hub_download

        kw = {"token": self._token} if self._token else {}
        ckpt = hf_hub_download(self.model_repo, "ft_model.pth", **kw)
        cfg = hf_hub_download(
            self.model_repo,
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
            **kw,
        )
        ref = hf_hub_download(self.model_repo, "ref_zephyr.wav", **kw)
        return ckpt, cfg, ref

    def load(self) -> None:
        """Download assets and load the model. Idempotent."""
        if self._loaded:
            return

        ckpt, cfg, ref = self._download_assets()
        self._reference_path = Path(ref)

        # Seed-VC's modules import relative to its own root.
        seed_dir = str(self._seed_dir)
        if seed_dir not in sys.path:
            sys.path.insert(0, seed_dir)
        cwd = os.getcwd()
        os.chdir(seed_dir)
        try:
            import inference as seedvc  # type: ignore

            args = self._make_args(
                source="", target=str(ref), output="", checkpoint=ckpt, config=cfg
            )
            log.info("Loading Seed-VC checkpoint (once)")
            models = seedvc.load_models(args)

            # Reuse the loaded models for every subsequent main() call.
            seedvc.load_models = lambda _args, _m=models: _m
            self._inference = seedvc
            self._ckpt, self._cfg = ckpt, cfg
        finally:
            os.chdir(cwd)
        self._loaded = True

    def _make_args(self, source, target, output, checkpoint, config) -> Namespace:
        return Namespace(
            source=source,
            target=target,
            output=output,
            diffusion_steps=self.diffusion_steps,
            length_adjust=self.length_adjust,
            inference_cfg_rate=self.cfg_rate,
            f0_condition=False,
            auto_f0_adjust=False,
            semi_tone_shift=0,
            checkpoint=checkpoint,
            config=config,
            fp16=self.fp16,
        )

    # -- conversion --------------------------------------------------------
    def convert_file(self, path: str | os.PathLike) -> tuple[np.ndarray, int]:
        """Convert a wav file. Returns ``(waveform, sample_rate)``."""
        import soundfile as sf

        self.load()
        assert self._inference is not None and self._reference_path is not None

        cwd = os.getcwd()
        os.chdir(self._seed_dir)
        tmp = tempfile.mkdtemp(prefix="ghana-vc-")
        try:
            args = self._make_args(
                source=str(path),
                target=str(self._reference_path),
                output=tmp,
                checkpoint=self._ckpt,
                config=self._cfg,
            )
            self._inference.main(args)
            produced = sorted(Path(tmp).glob("*.wav"))
            if not produced:
                raise RuntimeError(f"Seed-VC produced no output for {path}")
            audio, sr = sf.read(produced[0], dtype="float32")
            return audio, sr
        finally:
            os.chdir(cwd)
            shutil.rmtree(tmp, ignore_errors=True)

    def convert_array(self, audio: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, int]:
        """Convert an in-memory waveform. Returns ``(waveform, sample_rate)``."""
        import soundfile as sf

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:  # downmix to mono
            audio = audio.mean(axis=1)

        tmp = tempfile.mkdtemp(prefix="ghana-vc-src-")
        try:
            src = Path(tmp) / "source.wav"
            sf.write(src, audio, sampling_rate)
            return self.convert_file(src)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
