#!/usr/bin/env python3
"""Generate ghana-synthetic-speech dataset for Ghanaian languages using Gemini Live.

Extracts text from `ghananlpcommunity/ghana-speech`, normalises to universal
graphemes via `africa-g2p`, synthesises 24 kHz speech using Gemini Live (voice Zephyr),
packages into viewer-ready Parquet shards with embedded audio, and pushes to
`ghanaopenai/ghana-synthetic-speech`.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

# Ensure local .env is loaded
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ghana-synth")

# 42 language subsets available in ghananlpcommunity/ghana-speech
GHANA_SPEECH_CONFIGS: Dict[str, dict] = {
    "Akuapem_Twi_twi": {"name": "Akuapem Twi", "g2p": "twi", "iso": "twi"},
    "Anyin_any": {"name": "Anyin", "g2p": "any", "iso": "any"},
    "Asante_Twi_twi": {"name": "Asante Twi", "g2p": "twi", "iso": "twi"},
    "Avatime_avn": {"name": "Avatime", "g2p": "avn", "iso": "avn"},
    "Bassar_Ntcham_bud": {"name": "Ntcham", "g2p": "bud", "iso": "bud"},
    "Bimoba_bim": {"name": "Bimoba", "g2p": "bim", "iso": "bim"},
    "Birifor_Southern_biv": {"name": "Southern Birifor", "g2p": "biv", "iso": "biv"},
    "Bissa_bib": {"name": "Bissa", "g2p": "bib", "iso": "bib"},
    "Buli_bwu": {"name": "Buli", "g2p": "bwu", "iso": "bwu"},
    "Chumburung_ncu": {"name": "Chumburung", "g2p": "ncu", "iso": "ncu"},
    "Dagaare_dga": {"name": "Dagaare", "g2p": "dga", "iso": "dga"},
    "Dagbani_dag": {"name": "Dagbani", "g2p": "dag", "iso": "dag"},
    "Dangme_ada": {"name": "Dangme", "g2p": "ada", "iso": "ada"},
    "Deg_mzw": {"name": "Deg", "g2p": "mzw", "iso": "mzw"},
    "Ewe_ewe": {"name": "Ewe", "g2p": "ewe", "iso": "ewe"},
    "Fante_fat": {"name": "Fante", "g2p": "fat", "iso": "fat"},
    "Fulfulde_Maasina_ffm": {"name": "Fulfulde", "g2p": "ffm", "iso": "ffm"},
    "Gikyode_acd": {"name": "Gikyode", "g2p": "acd", "iso": "acd"},
    "Gonja_gjn": {"name": "Gonja", "g2p": "gjn", "iso": "gjn"},
    "Hausa_hau": {"name": "Hausa", "g2p": "hau", "iso": "hau"},
    "Kabiye_kbp": {"name": "Kabiye", "g2p": "kbp", "iso": "kbp"},
    "Kasem_xsm": {"name": "Kasem", "g2p": "xsm", "iso": "xsm"},
    "Konkomba_xon": {"name": "Konkomba", "g2p": "xon", "iso": "xon"},
    "Konni_kma": {"name": "Konni", "g2p": "kma", "iso": "kma"},
    "Kusaal_kus": {"name": "Kusaal", "g2p": "kus", "iso": "kus"},
    "Lelemi_lef": {"name": "Lelemi", "g2p": "lef", "iso": "lef"},
    "Mampruli_maw": {"name": "Mampruli", "g2p": "maw", "iso": "maw"},
    "Nawuri_naw": {"name": "Nawuri", "g2p": "naw", "iso": "naw"},
    "Ninkare_gur": {"name": "Frafra / Gurenne", "g2p": "gur", "iso": "gur"},
    "Nkonya_nko": {"name": "Nkonya", "g2p": "nko", "iso": "nko"},
    "Ntrubo_ntr": {"name": "Ntrubo", "g2p": "ntr", "iso": "ntr"},
    "Nzema_nzi": {"name": "Nzema", "g2p": "nzi", "iso": "nzi"},
    "Paasaal_sig": {"name": "Paasaal", "g2p": "sig", "iso": "sig"},
    "Sehwi_sfw": {"name": "Sehwi", "g2p": "sfw", "iso": "sfw"},
    "Sekpele_lip": {"name": "Sekpele", "g2p": "lip", "iso": "lip"},
    "Selee_snw": {"name": "Selee", "g2p": "snw", "iso": "snw"},
    "Sisaala_Tumulung_sil": {"name": "Sisaala", "g2p": "sil", "iso": "sil"},
    "Siwu_akp": {"name": "Siwu", "g2p": "akp", "iso": "akp"},
    "Tampulma_tpm": {"name": "Tampulma", "g2p": "tpm", "iso": "tpm"},
    "Tem_kdh": {"name": "Tem", "g2p": "kdh", "iso": "kdh"},
    "Tuwuli_bov": {"name": "Tuwuli", "g2p": "bov", "iso": "bov"},
    "Vagla_vag": {"name": "Vagla", "g2p": "vag", "iso": "vag"},
}

DEFAULT_WORK_DIR = (
    "/media/owusus/Godstestimo/NLP-Projects/ghana-synthetic-speech"
    if os.path.exists("/media/owusus/Godstestimo/NLP-Projects")
    else "./out/ghana-synthetic-speech"
)
DEFAULT_HF_REPO = "ghanaopenai/ghana-synthetic-speech"
DEFAULT_MODEL = "models/gemini-3.1-flash-live-preview"
DEFAULT_VOICE = "Zephyr"


# ---------------------------------------------------------------------------
# Stage 1: Text extraction from ghananlpcommunity/ghana-speech
# ---------------------------------------------------------------------------

def extract_source_sentences(
    config_name: str,
    target_count: int = 10000,
    min_chars: int = 20,
    max_chars: int = 240,
    save_path: Optional[str] = None,
) -> List[str]:
    """Fetch sentences from ghananlpcommunity/ghana-speech by reading only the text column."""
    if save_path and os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) >= target_count:
            log.info("Loaded %d existing sentences from %s", len(lines), save_path)
            return lines[:target_count]

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    pattern = f"datasets/ghananlpcommunity/ghana-speech/{config_name}/*.parquet"
    parquet_files = sorted(fs.glob(pattern))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found for {config_name} on HF: {pattern}")

    log.info(
        "Scanning %d parquet files for %s to gather up to %d sentences...",
        len(parquet_files),
        config_name,
        target_count,
    )

    collected: List[str] = []
    seen = set()

    for idx, path in enumerate(parquet_files):
        url = "hf://" + path
        try:
            with fsspec.open(url, "rb") as f:
                pf = pq.ParquetFile(f)
                tab = pf.read(columns=["text"])
                texts = tab["text"].to_pylist()
        except Exception as exc:
            log.warning("Failed to read %s: %s", url, exc)
            continue

        new_count = 0
        for text in texts:
            if not text:
                continue
            text = " ".join(str(text).split()).strip().strip('"').strip("'")
            if min_chars <= len(text) <= max_chars and text not in seen:
                seen.add(text)
                collected.append(text)
                new_count += 1
                if len(collected) >= target_count:
                    break

        log.info(
            "[%d/%d] %s: read %d rows, added %d valid sentences (total %d/%d)",
            idx + 1,
            len(parquet_files),
            os.path.basename(path),
            len(texts),
            new_count,
            len(collected),
            target_count,
        )

        if len(collected) >= target_count:
            break

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for text in collected:
                f.write(text + "\n")
        log.info("Wrote %d sentences to %s", len(collected), save_path)

    return collected


# ---------------------------------------------------------------------------
# Stage 2: Normalisation via africa-g2p
# ---------------------------------------------------------------------------

def normalise_sentences(
    sentences: Sequence[str],
    g2p_code: str,
    save_path: Optional[str] = None,
) -> List[str]:
    """Convert text sentences to Africa-G2P universal graphemes."""
    from afrispeech_synth.lang import Language
    from afrispeech_synth.normalise import Normaliser

    lang = Language(token=g2p_code, code=g2p_code, name=g2p_code, g2p_code=g2p_code)
    normaliser = Normaliser(lang, mode="universal")

    log.info("Normalising %d sentences to universal graphemes (lang=%s)...", len(sentences), g2p_code)
    normalised = [normaliser(text) for text in sentences]

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for text in normalised:
                f.write(text + "\n")
        log.info("Wrote %d normalised sentences to %s", len(normalised), save_path)

    return normalised


# ---------------------------------------------------------------------------
# Stage 3: Synthesis with Gemini Live (Multi-Key Pool Support)
# ---------------------------------------------------------------------------

class MultiKeyLiveWorkerPool:
    """Pool of Gemini Live backends across multiple API keys for 5x+ throughput."""

    def __init__(
        self,
        api_keys: List[str],
        config,
        language,
        per_key_concurrency: int = 4,
        per_key_rpm: int = 60,
    ):
        from google import genai
        from afrispeech_synth.synth import RateLimiter
        from afrispeech_synth.tts.gemini_live import GeminiLiveTTS

        self.api_keys = api_keys
        self.backends = []
        self.limiters = []
        self.semaphores = []
        self.per_key_concurrency = per_key_concurrency

        for key in api_keys:
            b = GeminiLiveTTS(config, language)
            b._client = genai.Client(api_key=key)
            self.backends.append(b)
            self.limiters.append(RateLimiter(per_key_rpm))
            self.semaphores.append(asyncio.Semaphore(per_key_concurrency))

    async def close(self):
        for b in self.backends:
            try:
                await b.close()
            except Exception:
                pass


async def synthesise_multi_key(
    utterances: Sequence,
    config,
    language,
    work_dir: str,
    api_keys: List[str],
    per_key_concurrency: int = 4,
    per_key_rpm: int = 60,
    resume: bool = True,
) -> dict:
    from afrispeech_synth.synth import Workspace, render_prompt, RetryableTTSError, TTSError

    workspace = Workspace(work_dir)
    pending = [u for u in utterances if not (resume and workspace.is_done(u))]
    skipped = len(utterances) - len(pending)
    if skipped:
        log.info("  resuming: %d already done, %d to go", skipped, len(pending))
    if not pending:
        return {"done": 0, "failed": 0, "skipped": skipped, "workspace": workspace}

    pool = MultiKeyLiveWorkerPool(
        api_keys=api_keys,
        config=config,
        language=language,
        per_key_concurrency=per_key_concurrency,
        per_key_rpm=per_key_rpm,
    )

    total_concurrency = len(api_keys) * per_key_concurrency
    total_rpm = len(api_keys) * per_key_rpm
    log.info(
        "  Multi-key pool active: %d API keys, total concurrency=%d, total RPM=%d",
        len(api_keys),
        total_concurrency,
        total_rpm,
    )

    queue = asyncio.Queue()
    for u in pending:
        queue.put_nowait(u)

    counters = {
        "done": 0,
        "failed": 0,
        "total": len(pending),
        "started": time.time(),
    }
    counter_lock = asyncio.Lock()

    async def worker(key_idx: int):
        backend = pool.backends[key_idx]
        limiter = pool.limiters[key_idx]
        semaphore = pool.semaphores[key_idx]

        while not queue.empty():
            try:
                utterance = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            async with semaphore:
                voice = utterance.voice or config.voice
                prompt = render_prompt(config, utterance.language or language, utterance.transcript)

                succeeded = False
                for attempt in range(1, config.max_retries + 1):
                    await limiter.acquire()
                    try:
                        clip = await backend.synth(prompt, voice)
                        path = workspace.audio_path(utterance)
                        with open(path, "wb") as handle:
                            handle.write(clip.audio)
                        workspace.write(utterance, {
                            "index": utterance.index,
                            "name": utterance.stem,
                            "language": (utterance.language or language).code,
                            "text": utterance.text,
                            "transcript": utterance.transcript,
                            "voice": clip.voice,
                            "audio_path": path,
                            "sample_rate": clip.sample_rate,
                            "bytes": len(clip.audio),
                        })
                        succeeded = True
                        break
                    except RetryableTTSError as exc:
                        if attempt == config.max_retries:
                            log.warning("[%d] giving up after %d tries: %s", utterance.index, attempt, exc)
                            break
                        delay = min(30.0, 2 ** attempt) * (0.5 + random.random())
                        await asyncio.sleep(delay)
                    except TTSError as exc:
                        log.warning("[%d] failed permanently: %s", utterance.index, exc)
                        break

            async with counter_lock:
                if succeeded:
                    counters["done"] += 1
                else:
                    counters["failed"] += 1

                done = counters["done"]
                total = counters["total"]
                if done % 50 == 0 or done == total:
                    elapsed = time.time() - counters["started"]
                    rate = (done / elapsed) * 60 if elapsed else 0
                    log.info(
                        "  %d/%d synthesised (%.0f clips/min, %d failed)",
                        done,
                        total,
                        rate,
                        counters["failed"],
                    )

            queue.task_done()

    tasks = []
    for key_idx in range(len(api_keys)):
        for _ in range(per_key_concurrency):
            tasks.append(asyncio.create_task(worker(key_idx)))

    try:
        await asyncio.gather(*tasks)
    finally:
        await pool.close()

    return {
        "done": counters["done"],
        "failed": counters["failed"],
        "skipped": skipped,
        "workspace": workspace,
    }


async def synthesise_subset(
    config_name: str,
    sentences: List[str],
    normalised_sentences: List[str],
    work_dir: str,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
    backend_name: str = "gemini-live",
    concurrency: int = 4,
    rpm: int = 60,
    sample_rate: int = 24000,
) -> dict:
    """Synthesise audio clips for a language subset using multi-key pool if available."""
    from afrispeech_synth.config import RunConfig
    from afrispeech_synth.lang import Language
    from afrispeech_synth.synth import Utterance, synthesise_async

    lang_meta = GHANA_SPEECH_CONFIGS[config_name]
    lang = Language(
        token=config_name,
        code=lang_meta["iso"],
        name=lang_meta["name"],
        g2p_code=lang_meta["g2p"],
    )

    run_cfg = RunConfig(language=lang_meta["iso"])
    run_cfg.tts.backend = backend_name
    run_cfg.tts.model = model
    run_cfg.tts.voice = voice
    run_cfg.tts.concurrency = concurrency
    run_cfg.tts.rpm = rpm
    run_cfg.tts.sample_rate = sample_rate

    utterances = [
        Utterance(index=i, text=raw, transcript=norm)
        for i, (raw, norm) in enumerate(zip(sentences, normalised_sentences))
    ]

    # Check for multiple keys
    raw_keys = os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "")
    api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    log.info(
        "Starting synthesis for %s: %d sentences (keys=%d, backend=%s, model=%s, voice=%s)",
        config_name,
        len(utterances),
        len(api_keys),
        backend_name,
        model,
        voice,
    )

    if len(api_keys) > 1 and backend_name == "gemini-live":
        return await synthesise_multi_key(
            utterances=utterances,
            config=run_cfg.tts,
            language=lang,
            work_dir=work_dir,
            api_keys=api_keys,
            per_key_concurrency=concurrency,
            per_key_rpm=rpm,
            resume=True,
        )

    result = await synthesise_async(
        utterances=utterances,
        config=run_cfg.tts,
        language=lang,
        work_dir=work_dir,
        resume=True,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 4: Package into Parquet shards
# ---------------------------------------------------------------------------

def package_subset(
    config_name: str,
    work_dir: str,
    out_dir: str,
    shard_target_mb: int = 190,
    sample_rate: int = 24000,
) -> List[str]:
    """Package synthesised clips into Parquet shards with embedded audio."""
    from afrispeech_synth.package import _shard_by_size, _hf_features_metadata, ROW_GROUP_ROWS
    from afrispeech_synth.synth import Workspace

    workspace = Workspace(work_dir)
    records = workspace.records()
    if not records:
        raise RuntimeError(f"No completed clips found in {work_dir}")

    log.info("Packaging %d clips for %s into %s", len(records), config_name, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    extra = ("language", "config_name")
    schema = pa.schema(
        [
            pa.field(
                "audio",
                pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
            ),
            pa.field("text", pa.string()),
            pa.field("normalised_text", pa.string()),
            pa.field("voice", pa.string()),
            pa.field("language", pa.string()),
            pa.field("config_name", pa.string()),
        ],
        metadata=_hf_features_metadata(sample_rate, extra),
    )

    shards = _shard_by_size(records, shard_target_mb * 1024 * 1024)
    log.info("%s: %d clips -> %d parquet shard(s)", config_name, len(records), len(shards))

    paths = []
    lang_iso = GHANA_SPEECH_CONFIGS[config_name]["iso"]

    for number, shard in enumerate(shards):
        name = f"train-{number:05d}-of-{len(shards):05d}.parquet"
        audio, text, normalised, voices, langs, cfgs = [], [], [], [], [], []

        for record in shard:
            with open(record["audio_path"], "rb") as handle:
                audio.append({
                    "bytes": handle.read(),
                    "path": os.path.basename(record["audio_path"]),
                })
            text.append(record["text"])
            normalised.append(record.get("transcript", record["text"]))
            voices.append(record.get("voice", ""))
            langs.append(lang_iso)
            cfgs.append(config_name)
            record["shard"] = name
            record["file_name"] = os.path.basename(record["audio_path"])

        table = pa.Table.from_pydict(
            {
                "audio": audio,
                "text": text,
                "normalised_text": normalised,
                "voice": voices,
                "language": langs,
                "config_name": cfgs,
            },
            schema=schema,
        )

        shard_path = os.path.join(out_dir, name)
        pq.write_table(
            table,
            shard_path,
            row_group_size=ROW_GROUP_ROWS,
            write_page_index=True,
        )
        paths.append(shard_path)
        log.info(
            "[%d/%d] %s (%d rows, %.1f MB)",
            number + 1,
            len(shards),
            shard_path,
            table.num_rows,
            os.path.getsize(shard_path) / 1e6,
        )

    # Write manifest metadata.jsonl
    manifest_path = os.path.join(out_dir, "metadata.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps({
                    "index": record["index"],
                    "text": record["text"],
                    "normalised_text": record.get("transcript", record["text"]),
                    "voice": record.get("voice", ""),
                    "shard": record.get("shard"),
                    "file_name": record.get("file_name") or os.path.basename(record["audio_path"]),
                    "config_name": config_name,
                    "language": lang_iso,
                }, ensure_ascii=False)
                + "\n"
            )

    log.info("Wrote manifest to %s", manifest_path)
    return paths


# ---------------------------------------------------------------------------
# Stage 5: Multi-subset Dataset Card and Hub Push
# ---------------------------------------------------------------------------

def generate_dataset_card(
    base_dir: str,
    active_configs: Sequence[str],
    repo_id: str = DEFAULT_HF_REPO,
    hub_configs: Optional[set] = None,
) -> str:
    """Generate root README.md with YAML dataset configs for Hugging Face.

    A config is listed in the ``configs:`` YAML if it has parquet shards locally
    (``base_dir``) OR already exists on the Hub (``hub_configs``).
    """
    configs_yaml = []
    summary_rows = []
    hub_configs = hub_configs or set()

    for cfg in sorted(active_configs):
        info = GHANA_SPEECH_CONFIGS[cfg]
        cfg_dir = os.path.join(base_dir, cfg)
        shards = sorted(Path(cfg_dir).glob("train-*.parquet")) if os.path.exists(cfg_dir) else []
        manifest = os.path.join(cfg_dir, "metadata.jsonl")
        num_clips = 0
        if os.path.exists(manifest):
            with open(manifest, "r", encoding="utf-8") as f:
                num_clips = sum(1 for _ in f)

        if shards or cfg in hub_configs:
            configs_yaml.append(f"""- config_name: {cfg}
  data_files:
  - split: train
    path: {cfg}/train-*""")

        if num_clips:
            status_str = f"**{num_clips:,} clips**"
        elif cfg in hub_configs:
            status_str = "**On Hub**"
        else:
            status_str = "Pending"
        summary_rows.append(f"| {info['name']} | `{cfg}` | `{info['iso']}` | {status_str} |")

    yaml_block = "\n".join(configs_yaml)
    table_block = "\n".join(summary_rows)

    card = f"""---
license: cc-by-nc-4.0
task_categories:
- text-to-speech
- automatic-speech-recognition
tags:
- audio
- tts
- voice-conversion
- seed-vc
- synthetic
- ghana
- african-languages
pretty_name: Ghana Synthetic Speech Dataset
configs:
{yaml_block}
---

# Ghana Synthetic Speech Dataset

A multi-subset synthetic speech dataset across **42 Ghanaian and West African languages**,
specifically created to fine-tune **[Seed-VC](https://github.com/Plachtaa/seed-vc)** and other
voice conversion / TTS models into a consistent Ghanaian voice (**Zephyr**).

Source text is drawn directly from [`ghananlpcommunity/ghana-speech`](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech)
and normalised to cross-lingual African universal graphemes with [`africa-g2p`](https://github.com/AfriSpeech/africa-g2p).
Speech is synthesised using **Gemini Live** (`models/gemini-3.1-flash-live-preview`), voice **Zephyr**, at 24 kHz mono WAV.

## Language Subsets

| Language | Config Name | ISO Code | Status |
|---|---|---|---|
{table_block}

## How to Load

Load any specific language subset using its config name:

```python
from datasets import load_dataset

# Load Dagbani subset
ds = load_dataset("{repo_id}", "Dagbani_dag", split="train")

# Access audio and normalised text
sample = ds[0]
print(sample["text"])
print(sample["normalised_text"])
audio_array = sample["audio"]["array"]
sampling_rate = sample["audio"]["sampling_rate"]  # 24,000 Hz
```

## Features

- **Embedded Audio**: 24,000 Hz mono PCM WAV embedded directly in Parquet shards. The dataset viewer plays each clip inline.
- **Consistent Target Speaker**: Synthesised with voice **Zephyr** across all languages, enabling direct multi-language voice conversion fine-tuning.
- **Universal Graphemes**: Canonical African grapheme conversion for orthographic consistency across phonologically related languages.

## Columns

| Column | Type | Description |
|---|---|---|
| `audio` | `Audio` | Synthesised speech (24 kHz mono WAV) |
| `text` | `string` | Original transcript from `ghana-speech` |
| `normalised_text` | `string` | Universal grapheme normalised transcript spoken by Gemini Live |
| `voice` | `string` | Speaker voice name (`Zephyr`) |
| `language` | `string` | ISO 639-3 language code |
| `config_name` | `string` | Subset config name |

## License & Attribution

- **License:** CC-BY-NC-4.0
- **Source Transcripts:** [`ghananlpcommunity/ghana-speech`](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech)
- **Normalisation:** [`africa-g2p`](https://github.com/AfriSpeech/africa-g2p)
- **Synthesis:** Google Gemini Live (`models/gemini-3.1-flash-live-preview`, voice `Zephyr`)
"""
    return card


def push_subset_to_hub(
    base_dir: str,
    config_name: str,
    repo_id: str = DEFAULT_HF_REPO,
    token: Optional[str] = None,
    private: bool = False,
) -> None:
    """Push a single completed subset and updated root README to Hugging Face."""
    from huggingface_hub import HfApi

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    cfg_dir = os.path.join(base_dir, config_name)
    if not os.path.exists(cfg_dir):
        raise RuntimeError(f"Subset directory {cfg_dir} does not exist.")

    log.info("Uploading %s to %s on the Hub (parquet shards + manifests)...", config_name, repo_id)
    api.upload_folder(
        folder_path=cfg_dir,
        path_in_repo=config_name,
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=["work/**"],
    )

    # Update root dataset README
    hub_configs = set()
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        hubs_c = set()
        for rf in repo_files:
            if "/" in rf and rf.endswith(".parquet"):
                hubs_c.add(rf.split("/")[0])
        hub_configs = hubs_c
    except Exception:
        hub_configs = {config_name}
    card_content = generate_dataset_card(
        base_dir, list(GHANA_SPEECH_CONFIGS.keys()), repo_id, hub_configs=hub_configs
    )
    readme_path = os.path.join(base_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card_content)

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    log.info("Subset %s successfully published to https://huggingface.co/datasets/%s", config_name, repo_id)


# ---------------------------------------------------------------------------
# Status Reporting
# ---------------------------------------------------------------------------

def print_status(base_dir: str) -> None:
    """Inspect and display status across all 42 language subsets."""
    from afrispeech_synth.synth import Workspace

    print("\n" + "=" * 80)
    print(f"{'GHANA SYNTHETIC SPEECH DATASET - STATUS':^80}")
    print(f"{base_dir:^80}")
    print("=" * 80)
    print(f"{'Config':<25} {'Language':<18} {'Sentences':<10} {'Synthesised':<13} {'Parquet':<10}")
    print("-" * 80)

    total_sentences = 0
    total_synth = 0

    for cfg, info in sorted(GHANA_SPEECH_CONFIGS.items()):
        cfg_dir = os.path.join(base_dir, cfg)
        sentences_file = os.path.join(cfg_dir, "sentences.txt")
        num_sentences = 0
        if os.path.exists(sentences_file):
            with open(sentences_file, "r", encoding="utf-8") as f:
                num_sentences = sum(1 for _ in f)

        work_dir = os.path.join(cfg_dir, "work")
        manifest_file = os.path.join(cfg_dir, "metadata.jsonl")
        num_synth = 0
        if os.path.exists(manifest_file):
            with open(manifest_file, "r", encoding="utf-8") as f:
                num_synth = sum(1 for _ in f)
        elif os.path.exists(work_dir):
            try:
                ws = Workspace(work_dir)
                num_synth = len(ws.records())
            except Exception:
                num_synth = 0

        parquet_files = list(Path(cfg_dir).glob("train-*.parquet")) if os.path.exists(cfg_dir) else []
        pq_str = f"{len(parquet_files)} shard(s)" if parquet_files else "none"

        total_sentences += num_sentences
        total_synth += num_synth

        status_tag = ""
        if num_sentences > 0 and num_synth >= num_sentences and parquet_files:
            status_tag = " [DONE]"

        print(
            f"{cfg:<25} {info['name']:<18} {num_sentences:<10} {num_synth:<13} {pq_str:<10}{status_tag}"
        )

    print("-" * 80)
    print(f"Total Sentences Extracted: {total_sentences:,}")
    print(f"Total Clips Synthesised:   {total_synth:,}")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Master Runner
# ---------------------------------------------------------------------------

async def run_pipeline(
    subsets: List[str],
    work_dir: str,
    max_sentences: int,
    voice: str,
    model: str,
    backend: str,
    concurrency: int,
    rpm: int,
    stage: str,
    push_to: Optional[str],
    token: Optional[str],
    cleanup: str = "work",
) -> None:
    """Execute the pipeline across the given subsets."""
    log.info("Processing %d subset(s) in %s (stage=%s)", len(subsets), work_dir, stage)

    for idx, cfg in enumerate(subsets):
        log.info("\n" + "=" * 70)
        log.info("[%d/%d] SUBSET: %s (%s)", idx + 1, len(subsets), cfg, GHANA_SPEECH_CONFIGS[cfg]["name"])
        log.info("=" * 70)

        cfg_dir = os.path.join(work_dir, cfg)
        existing_shards = list(Path(cfg_dir).glob("train-*.parquet")) if os.path.exists(cfg_dir) else []
        manifest_file = os.path.join(cfg_dir, "metadata.jsonl")

        # Check if already completed locally or on Hugging Face
        hub_completed = False
        if push_to:
            try:
                from huggingface_hub import HfApi
                api_check = HfApi(token=token or os.environ.get("HF_TOKEN"))
                hub_files = api_check.list_repo_files(push_to, repo_type="dataset")
                hub_shards = [f for f in hub_files if f.startswith(f"{cfg}/train-") and f.endswith(".parquet")]
                if len(hub_shards) >= 5: # Completed subset has multiple shards
                    hub_completed = True
            except Exception:
                pass

        if stage == "all" and ((existing_shards and os.path.exists(manifest_file)) or hub_completed):
            log.info("Subset %s is ALREADY COMPLETED (verified on Hub or local). Skipping!", cfg)
            continue

        synth_work_dir = os.path.join(cfg_dir, "work")
        sentences_file = os.path.join(cfg_dir, "sentences.txt")
        norm_file = os.path.join(cfg_dir, "normalised_sentences.txt")

        # 1. Extraction
        if stage in ("all", "extract"):
            sentences = extract_source_sentences(
                config_name=cfg,
                target_count=max_sentences,
                save_path=sentences_file,
            )
        else:
            if not os.path.exists(sentences_file):
                log.error("Sentences file %s missing. Run with --stage extract first.", sentences_file)
                continue
            with open(sentences_file, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()][:max_sentences]

        # 2. Normalisation
        g2p_code = GHANA_SPEECH_CONFIGS[cfg]["g2p"]
        if stage in ("all", "extract") or not os.path.exists(norm_file):
            normalised = normalise_sentences(sentences, g2p_code=g2p_code, save_path=norm_file)
        else:
            with open(norm_file, "r", encoding="utf-8") as f:
                normalised = [line.strip() for line in f if line.strip()][:len(sentences)]

        # 3. Synthesis
        if stage in ("all", "synth"):
            await synthesise_subset(
                config_name=cfg,
                sentences=sentences,
                normalised_sentences=normalised,
                work_dir=synth_work_dir,
                voice=voice,
                model=model,
                backend_name=backend,
                concurrency=concurrency,
                rpm=rpm,
            )

        # 4. Packaging
        if stage in ("all", "package"):
            package_subset(
                config_name=cfg,
                work_dir=synth_work_dir,
                out_dir=cfg_dir,
            )

        # 5. Push to Hub
        if stage in ("all", "push") and push_to:
            push_subset_to_hub(
                base_dir=work_dir,
                config_name=cfg,
                repo_id=push_to,
                token=token,
            )
            if cleanup in ("work", "all") and os.path.exists(synth_work_dir):
                log.info("Deleting raw work directory %s to preserve space...", synth_work_dir)
                shutil.rmtree(synth_work_dir, ignore_errors=True)
            if cleanup == "all":
                for pq_path in Path(cfg_dir).glob("train-*.parquet"):
                    pq_path.unlink(missing_ok=True)
                log.info("Removed local parquet shards for %s after push to Hub", cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate_ghana_synthetic_speech",
        description="Synthesise multi-lingual Ghanaian speech datasets for Seed-VC fine-tuning.",
    )
    parser.add_argument(
        "--subset",
        default="all",
        help="Subset name (e.g. Dagbani_dag) or comma-separated list, or 'all' for all 42 subsets.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=5000,
        help="Target number of sentences per subset (default: 5,000).",
    )
    parser.add_argument(
        "--work-dir",
        default=DEFAULT_WORK_DIR,
        help=f"Base output directory (default: {DEFAULT_WORK_DIR}).",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Gemini voice name (default: {DEFAULT_VOICE}).",
    )
    parser.add_argument(
        "--backend",
        default="gemini-live",
        choices=["gemini-live", "gemini"],
        help="TTS backend (default: gemini-live).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model identifier (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent websocket sessions (default: 4, safe for Live API quota).",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=60,
        help="Rate limit: requests per minute (default: 60).",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "extract", "synth", "package", "push"],
        help="Run specific stage or 'all'.",
    )
    parser.add_argument(
        "--push-to",
        default=DEFAULT_HF_REPO,
        help=f"Hugging Face repo ID to push to (default: {DEFAULT_HF_REPO}). Set to '' to skip push.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token (falls back to HF_TOKEN or cached login).",
    )
    parser.add_argument(
        "--cleanup",
        default="work",
        choices=["work", "all", "none"],
        help="Space preservation: 'work' deletes raw wav/json clips after push, 'all' deletes local parquet too, 'none' keeps everything.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display summary status across all 42 language subsets and exit.",
    )

    args = parser.parse_args()

    if args.status:
        print_status(args.work_dir)
        return

    # Determine which subsets to run
    if args.subset.lower() == "all":
        subsets = list(GHANA_SPEECH_CONFIGS.keys())
    else:
        requested = [s.strip() for s in args.subset.split(",") if s.strip()]
        subsets = []
        for req in requested:
            if req in GHANA_SPEECH_CONFIGS:
                subsets.append(req)
            else:
                matches = [k for k in GHANA_SPEECH_CONFIGS if req.lower() in k.lower()]
                if matches:
                    subsets.extend(matches)
                else:
                    log.error("Unknown subset: %s. Available: %s", req, list(GHANA_SPEECH_CONFIGS.keys()))
                    sys.exit(1)

    asyncio.run(
        run_pipeline(
            subsets=subsets,
            work_dir=args.work_dir,
            max_sentences=args.max_sentences,
            voice=args.voice,
            model=args.model,
            backend=args.backend,
            concurrency=args.concurrency,
            rpm=args.rpm,
            stage=args.stage,
            push_to=args.push_to if args.push_to else None,
            token=args.token,
            cleanup=args.cleanup,
        )
    )


if __name__ == "__main__":
    main()
