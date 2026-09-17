"""Command line interface for twi-zephyr-vc."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .config import (
    ALLOWED_DIFFUSION_STEPS,
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_MODEL_REPO,
)


def _steps(value: str) -> int:
    steps = int(value)
    if steps < 1:
        raise argparse.ArgumentTypeError("--diffusion-steps must be >= 1")
    if steps not in ALLOWED_DIFFUSION_STEPS:
        print(
            f"note: --diffusion-steps {steps} is outside the tested values "
            f"{ALLOWED_DIFFUSION_STEPS}; 50 is recommended.",
            file=sys.stderr,
        )
    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twi-zephyr-vc",
        description="Convert Hugging Face audio datasets into the Twi Zephyr voice.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("convert", help="convert a dataset and push it to the Hub")
    conv.add_argument("--dataset", required=True, help="source dataset, e.g. org/name")
    conv.add_argument("--output", required=True, help="destination dataset, e.g. org/name-zephyr")
    conv.add_argument("--split", default="train")
    conv.add_argument("--config-name", default=None, help="dataset config name")
    conv.add_argument("--audio-column", default=None,
                      help="audio column (auto-detected when omitted)")
    conv.add_argument("--num-samples", type=int, default=None,
                      help="convert only the first N samples")
    conv.add_argument("--diffusion-steps", type=_steps, default=DEFAULT_DIFFUSION_STEPS,
                      help="25 (fast, robotic), 50 (recommended), 100 (slowest)")
    conv.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    conv.add_argument("--token", default=None,
                      help="HF token (falls back to $HF_TOKEN, then cached login)")
    conv.add_argument("--private", action="store_true", help="push as a private dataset")
    conv.add_argument("--drop-original", action="store_true",
                      help="remove the source audio column from the output")
    conv.add_argument("--no-install-deps", action="store_true",
                      help="skip installing Seed-VC requirements")
    conv.add_argument("-v", "--verbose", action="store_true")

    one = sub.add_parser("convert-file", help="convert a single audio file")
    one.add_argument("source", help="input audio file")
    one.add_argument("-o", "--output", default="converted.wav")
    one.add_argument("--diffusion-steps", type=_steps, default=DEFAULT_DIFFUSION_STEPS)
    one.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    one.add_argument("--token", default=None)
    one.add_argument("--no-install-deps", action="store_true")
    one.add_argument("-v", "--verbose", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    token = args.token or os.environ.get("HF_TOKEN")

    if args.command == "convert":
        from .dataset import convert_dataset

        convert_dataset(
            args.dataset,
            args.output,
            split=args.split,
            config_name=args.config_name,
            audio_column=args.audio_column,
            num_samples=args.num_samples,
            diffusion_steps=args.diffusion_steps,
            model_repo=args.model_repo,
            token=token,
            private=args.private,
            keep_original=not args.drop_original,
            install_deps=not args.no_install_deps,
        )
        return 0

    if args.command == "convert-file":
        import soundfile as sf

        from .engine import ZephyrConverter

        conv = ZephyrConverter(
            diffusion_steps=args.diffusion_steps,
            model_repo=args.model_repo,
            token=token,
            install_deps=not args.no_install_deps,
        )
        wav, sr = conv.convert_file(args.source)
        sf.write(args.output, wav, sr)
        print(f"wrote {args.output} ({len(wav)/sr:.2f}s @ {sr} Hz)")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
