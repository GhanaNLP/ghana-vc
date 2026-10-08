"""Fine-tune a per-language Seed-VC model for each synthetic-speech config.

One model per language, pushed to ``ghanaopenai/ghana-vc-<config>`` and grouped
into a Hugging Face Collection. Each model starts from the zero-shot
``seed-uvit-whisper-small-wavenet`` base and is trained only on that language's
Zephyr-voiced clips, so converting audio in a language uses the model that
heard that language.

Runs against a local checkout of Seed-VC with its own venv::

    python scripts/finetune_languages.py \\
      --data-dir  /mnt/volume_d2wey28/data/ghana-synthetic-speech \\
      --seedvc-dir /mnt/volume_d2wey28/projects/ghana-vc/seed-vc \\
      --venv-python /mnt/volume_d2wey28/projects/ghana-vc/.venv-seedvc/bin/python \\
      --work-dir /mnt/volume_d2wey28/data/ghana-vc-fin \\
      --steps 1250 --configs all

    # a single language:
      --configs Akuapem_Twi_twi

Resumable: a config is skipped when its model repo already exists on the Hub
(``--force``) or its ft_model.pth is already produced locally.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("finetune_languages")

# Mirrors the pipeline's config map.
CONFIGS = {
    "Akuapem_Twi_twi": {"name": "Akuapem Twi", "iso": "twi"},
    "Anyin_any": {"name": "Anyin", "iso": "any"},
    "Asante_Twi_twi": {"name": "Asante Twi", "iso": "twi"},
    "Avatime_avn": {"name": "Avatime", "iso": "avn"},
    "Bassar_Ntcham_bud": {"name": "Ntcham", "iso": "bud"},
    "Bimoba_bim": {"name": "Bimoba", "iso": "bim"},
    "Birifor_Southern_biv": {"name": "Southern Birifor", "iso": "biv"},
    "Bissa_bib": {"name": "Bissa", "iso": "bib"},
    "Buli_bwu": {"name": "Buli", "iso": "bwu"},
    "Chumburung_ncu": {"name": "Chumburung", "iso": "ncu"},
    "Dagaare_dga": {"name": "Dagaare", "iso": "dga"},
    "Dagbani_dag": {"name": "Dagbani", "iso": "dag"},
    "Dangme_ada": {"name": "Dangme", "iso": "ada"},
    "Deg_mzw": {"name": "Deg", "iso": "mzw"},
    "Ewe_ewe": {"name": "Ewe", "iso": "ewe"},
    "Fante_fat": {"name": "Fante", "iso": "fat"},
    "Fulfulde_Maasina_ffm": {"name": "Fulfulde", "iso": "ffm"},
    "Gikyode_acd": {"name": "Gikyode", "iso": "acd"},
    "Gonja_gjn": {"name": "Gonja", "iso": "gjn"},
    "Hausa_hau": {"name": "Hausa", "iso": "hau"},
    "Kabiye_kbp": {"name": "Kabiye", "iso": "kbp"},
    "Kasem_xsm": {"name": "Kasem", "iso": "xsm"},
    "Konkomba_xon": {"name": "Konkomba", "iso": "xon"},
    "Konni_kma": {"name": "Konni", "iso": "kma"},
    "Kusaal_kus": {"name": "Kusaal", "iso": "kus"},
    "Lelemi_lef": {"name": "Lelemi", "iso": "lef"},
    "Mampruli_maw": {"name": "Mampruli", "iso": "maw"},
    "Nawuri_naw": {"name": "Nawuri", "iso": "naw"},
    "Ninkare_gur": {"name": "Frafra / Gurenne", "iso": "gur"},
    "Nkonya_nko": {"name": "Nkonya", "iso": "nko"},
    "Ntrubo_ntr": {"name": "Ntrubo", "iso": "ntr"},
    "Nzema_nzi": {"name": "Nzema", "iso": "nzi"},
    "Paasaal_sig": {"name": "Paasaal", "iso": "sig"},
    "Sehwi_sfw": {"name": "Sehwi", "iso": "sfw"},
    "Sekpele_lip": {"name": "Sekpele", "iso": "lip"},
    "Selee_snw": {"name": "Selee", "iso": "snw"},
    "Sisaala_Tumulung_sil": {"name": "Sisaala", "iso": "sil"},
    "Siwu_akp": {"name": "Siwu", "iso": "akp"},
    "Tampulma_tpm": {"name": "Tampulma", "iso": "tpm"},
    "Tem_kdh": {"name": "Tem", "iso": "kdh"},
    "Tuwuli_bov": {"name": "Tuwuli", "iso": "bov"},
    "Vagla_vag": {"name": "Vagla", "iso": "vag"},
}

BASE_MODEL_REPO = "Plachta/Seed-VC"
BASE_CKPT = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
# The fine-tune config pushed with the existing Twi model; matches the base arch.
FT_CONFIG = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"

MODEL_ORG = "ghanaopenai"
COLLECTION_TITLE = "ghana-vc-models"
LICENSE_URL = "https://raw.githubusercontent.com/GhanaNLP/ghana-vc/main/LICENSE"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fine-tune Seed-VC per language from synthetic speech."
    )
    p.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="one or more config names, or 'all'",
    )
    p.add_argument("--data-dir", required=True, help="dir with <config>/*.parquet")
    p.add_argument("--seedvc-dir", required=True, help="checked-out Seed-VC repo")
    p.add_argument(
        "--venv-python",
        required=True,
        help="python from the venv that has Seed-VC's torch/deps",
    )
    p.add_argument("--work-dir", required=True, help="scratch + staged models")
    # Step 1000 kept pronunciation clearest in a 500-3500 checkpoint comparison.
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--push-to", default=f"{MODEL_ORG}", help="model org")
    p.add_argument("--token", default=None, help="HF token (env HF_TOKEN used if empty)")
    p.add_argument("--force", action="store_true", help="re-train even if repo exists")
    p.add_argument("--no-push", action="store_true", help="train+stage only, no Hub write")
    p.add_argument("--keep-wavs", action="store_true", help="don't delete staged wavs")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def each_config(names):
    if names == ["all"]:
        names = list(CONFIGS)
    bad = [n for n in names if n not in CONFIGS]
    if bad:
        raise SystemExit(f"unknown config(s): {bad}")
    return names


# -- data staging -----------------------------------------------------------

def extract_wavs(config, data_dir, wav_dir: Path) -> int:
    """Copy wav bytes out of the packaged parquet shards. Returns clip count."""
    import pyarrow.parquet as pq

    if wav_dir.exists() and any(wav_dir.glob("*.wav")):
        return len(list(wav_dir.glob("*.wav")))

    wav_dir.mkdir(parents=True, exist_ok=True)
    # Fill in any shards missing locally (a partial local copy would silently
    # train on a fraction of the data).
    from huggingface_hub import HfApi, hf_hub_download

    repo = "ghanaopenai/ghana-synthetic-speech"
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    remote_shards = [f for f in files
                     if f.startswith(f"{config}/train-") and f.endswith(".parquet")]
    missing = [f for f in remote_shards if not (Path(data_dir) / f).exists()]
    if missing:
        log.info("%s: downloading %d/%d missing shard(s) from the Hub",
                 config, len(missing), len(remote_shards))
        for f in missing:
            hf_hub_download(repo, f, repo_type="dataset", local_dir=data_dir)
    shards = sorted(Path(data_dir).glob(f"{config}/train-*.parquet"))
    if not shards:
        raise RuntimeError(f"no parquet shards for {config} in {data_dir} or on Hub")

    count = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["audio"])
        for row in table.to_pylist():
            if not row["audio"] or not row["audio"].get("bytes"):
                continue
            path = row["audio"].get("path") or f"utt_{count:06d}.wav"
            (wav_dir / Path(path).name).write_bytes(row["audio"]["bytes"])
            count += 1
    log.info("%s: staged %d wavs -> %s", config, count, wav_dir)
    return count


# -- training ---------------------------------------------------------------

def train_config(config, args, base_ckpt: Path, model_cfg: Path):
    wav_dir = Path(args.work_dir) / "wavs" / config
    extract_wavs(config, args.data_dir, wav_dir)

    # train.py writes to config's log_dir + run_name, relative to its CWD.
    cmd = [
        args.venv_python,
        "train.py",
        "--config", str(model_cfg),
        "--pretrained-ckpt", str(base_ckpt),
        "--dataset-dir", str(wav_dir),
        "--run-name", config,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--max-steps", str(args.steps),
        "--save-every", str(args.save_every),
        "--gpu", str(args.gpu),
    ]
    log.info("Running %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=args.seedvc_dir)
    if result.returncode != 0:
        raise RuntimeError(f"train.py failed for {config} (rc={result.returncode})")


def trained_ft_path(seedvc_dir: Path, config: str) -> Path | None:
    # log_dir is "./runs" in the fine-tune config; train.py appends run_name.
    p = seedvc_dir / "runs" / config / "ft_model.pth"
    return p if p.exists() else None


# -- Hub push ---------------------------------------------------------------

def stage_and_push(config, args, ft_model: Path, model_cfg: Path):
    from huggingface_hub import upload_folder

    push_to = args.push_to or MODEL_ORG
    repo_id = f"{push_to}/ghana-vc-{config}"

    stage = Path(args.work_dir) / "staged" / config
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # model + config + language reference clip (only the final weights; the
    # intermediate DiT_epoch_* checkpoints carry optimizer state and stay local)
    shutil.copy2(ft_model, stage / "ft_model.pth")
    shutil.copy2(model_cfg, stage / FT_CONFIG)

    # reference: the first Zephyr clip of the language, pulled from the parquet
    wav_dir = Path(args.work_dir) / "wavs" / config
    refs = sorted(wav_dir.glob("*.wav"))
    if not refs:
        raise RuntimeError(f"no staged wavs for reference: {config}")
    shutil.copy2(refs[0], stage / "ref_zephyr.wav")

    if args.no_push:
        log.info("[no-push] staged %s at %s (would push %s)", config, stage, repo_id)
        return repo_id, stage

    # license text (GPL-3.0 upstream Seed-VC attribution, matches existing model)
    lic_path = stage / "LICENSE"
    if not lic_path.exists():
        import urllib.request

        urllib.request.urlretrieve(LICENSE_URL, lic_path)

    (stage / "README.md").write_text(model_readme(config), encoding="utf-8")

    from huggingface_hub import create_repo

    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False, token=args.token)
    upload_folder(repo_id=repo_id, folder_path=str(stage), repo_type="model",
                  token=args.token)
    log.info("pushed %s", repo_id)

    # group into the collection
    from huggingface_hub import add_collection_item, create_collection, get_collection

    slug = f"{push_to}/{COLLECTION_TITLE}"
    try:
        try:
            coll = get_collection(slug, token=args.token)
        except Exception:
            coll = create_collection(
                title=COLLECTION_TITLE,
                namespace=push_to,
                description=(
                    f"{len(CONFIGS)} per-language ghana-vc models, one for each "
                    "Ghanaian-language voice available in ghanaopenai/ghana-synthetic-speech."
                ),
                token=args.token,
            )
        add_collection_item(
            coll.slug,
            item_id=repo_id,
            item_type="model",
            exists_ok=True,
            token=args.token,
        )
        log.info("added %s to collection %s", repo_id, coll.slug)
    except Exception as exc:
        log.warning("could not add %s to collection: %s", repo_id, exc)
    return repo_id, stage


def model_readme(config):
    meta = CONFIGS[config]
    iso = meta["iso"]
    return f"""---
license: cc-by-nc-4.0
pipeline_tag: audio-to-audio
language:
  - {iso}
tags:
  - audio
  - voice-conversion
  - seed-vc
  - {iso}
  - ghana
---

# ghana-vc-{config}

Voice conversion for **{meta['name']}** ({iso}). An audio-to-audio model that
converts speech into the *Zephyr* voice while preserving the language.

This checkpoint is dedicated to {meta['name']}: it was fine-tuned only on
{meta['name']} ({iso}) clips from
[ghanaopenai/ghana-synthetic-speech](https://huggingface.co/datasets/ghanaopenai/ghana-synthetic-speech),
starting from the zero-shot
[seed-uvit-whisper-small-wavenet](https://huggingface.co/Plachta/Seed-VC) base.
Use it to convert audio in {meta['name']}. For other languages — or as a
cross-lingual fallback — use [ghanaopenai/ghana-vc](https://huggingface.co/ghanaopenai/ghana-vc).

## Use with ghana-vc

```bash
pip install git+https://github.com/GhanaNLP/ghana-vc

export HF_TOKEN=hf_...
ghana-vc convert \\
  --dataset <org>/<audio-dataset> \\
  --output  <org>/<result> \\
  --config-name {config} \\
  --num-samples 100
```

The library auto-selects this model when the dataset/config name matches.

## Files

| File | Description |
| --- | --- |
| `ft_model.pth` | Fine-tuned DiT weights (step 1,000) |
| `{FT_CONFIG}` | Model / training config |
| `ref_zephyr.wav` | Reference utterance of the *Zephyr* voice in {meta['name']} |
| `LICENSE` | Upstream Seed-VC GPL-3.0 text, retained for attribution |

## Training

- Dataset: `ghanaopenai/ghana-synthetic-speech` config `{config}` ({meta['name']})
- Base: `seed-uvit-whisper-small-wavenet` (zero-shot)
- Weights: the step-1,000 checkpoint. Comparing checkpoints from 500 to 3,500
  steps on Asante Twi, step 1,000 kept pronunciation clearest (lowest character
  error rate with Omnilingual ASR CTC-300M), so it is the one published.
"""


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args.token = args.token or os.environ.get("HF_TOKEN")

    configs = each_config(args.configs)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    base_ckpt = Path(hf_hub_download(BASE_MODEL_REPO, BASE_CKPT,
                                     cache_dir=Path(args.work_dir) / "cache", token=args.token))
    model_cfg = Path(hf_hub_download(f"{MODEL_ORG}/ghana-vc", FT_CONFIG,
                                     cache_dir=Path(args.work_dir) / "cache", token=args.token))
    log.info("base checkpoint: %s", base_ckpt)
    log.info("fine-tune config: %s", model_cfg)

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)

    failed = []
    for i, config in enumerate(configs, 1):
        repo_id = f"{args.push_to or MODEL_ORG}/ghana-vc-{config}"
        log.info("[%d/%d] ==== %s ====", i, len(configs), config)

        if not args.no_push and not args.force:
            try:
                api.model_info(repo_id)
                log.info("already on the Hub, skipping: %s", repo_id)
                continue
            except Exception:
                pass

        try:
            # Use a local checkpoint if present, else re-train.
            ft = trained_ft_path(Path(args.seedvc_dir), config)
            if ft is None or args.force:
                train_config(config, args, base_ckpt, model_cfg)
                ft = trained_ft_path(Path(args.seedvc_dir), config)
                if ft is None:
                    raise RuntimeError(f"training done but no ft_model.pth for {config}")

            stage_and_push(config, args, ft, model_cfg)
        except Exception:
            # One bad language shouldn't stop a 40-language overnight run.
            log.exception("FAILED %s, moving on", config)
            failed.append(config)

        # Clean up staged files and run checkpoints (pushed, or a failed run's
        # leftovers) so 40 languages don't fill the disk.
        if not args.no_push:
            stage_dir = Path(args.work_dir) / "staged" / config
            shutil.rmtree(stage_dir, ignore_errors=True)
            run_dir = Path(args.seedvc_dir) / "runs" / config
            shutil.rmtree(run_dir, ignore_errors=True)

        if not args.keep_wavs:
            wav_dir = Path(args.work_dir) / "wavs"
            shutil.rmtree(wav_dir, ignore_errors=True)

    log.info("Done: %d language model(s), %d failed.", len(configs) - len(failed), len(failed))
    if failed:
        log.error("Failed: %s", ", ".join(failed))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())