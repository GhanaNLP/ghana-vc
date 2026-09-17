# twi-zephyr-vc

Convert **any** Hugging Face audio dataset into the Twi *Zephyr* voice, and push
the result back to the Hub with the converted audio as an added column.

The conversion is **cross-lingual** — the source speech does not have to be Twi.
It runs on [Seed-VC](https://github.com/Plachtaa/seed-vc) with the
[`ghanaopenai/twi-zephyr-vc`](https://huggingface.co/ghanaopenai/twi-zephyr-vc)
checkpoint.

## Install

```bash
pip install git+https://github.com/GhanaNLP/twi-zephyr-vc
```

Not yet on PyPI — install from source as above.

Seed-VC itself is not on PyPI, so it is cloned into `~/.cache/twi-zephyr-vc/`
on first use and its requirements installed. Set `TWI_ZEPHYR_VC_HOME` to change
that location, or pass `--no-install-deps` if you manage the environment
yourself.

## Hardware

**A GPU is strongly recommended.** Seed-VC falls back to CPU automatically but
is far too slow there for anything beyond a handful of clips.

| | |
| --- | --- |
| Minimum useful GPU | NVIDIA T4 (16 GB) — the cheapest cloud tier, sufficient |
| VRAM needed | ~4-6 GB (412 MB checkpoint + Whisper-small + BigVGAN) |
| Faster options | L4, A10G, L40S — all comfortably exceed requirements |
| CPU | Works, but expect many times slower; fine for one-off files |

There is no benefit to a large GPU here: inference is one clip at a time and is
not batched, so an A100 or H200 will not be meaningfully faster than a T4.
Pick the cheapest GPU available.

## Use

### A Hugging Face dataset

Point it at a dataset, give it somewhere to push:

```bash
export HF_TOKEN=hf_...

twi-zephyr-vc convert \
  --dataset mozilla-foundation/common_voice_17_0 \
  --output my-org/common-voice-zephyr \
  --num-samples 100
```

That converts the first 100 samples at the default 50 diffusion steps and pushes
the result. Every original column is preserved; three are added:

| Column | Contents |
| --- | --- |
| `audio_zephyr` | the converted audio |
| `zephyr_diffusion_steps` | steps used, for provenance |
| `zephyr_model` | checkpoint used, for provenance |

### Options

```
--dataset            source dataset (org/name)            [required]
--output             destination dataset (org/name)       [required]
--split              default: train
--config-name        dataset config, when it has several
--audio-column       auto-detected when omitted
--num-samples        convert only the first N
--diffusion-steps    25 | 50 | 100        (default: 50)
--token              HF token; falls back to $HF_TOKEN, then cached login
--private            push as a private dataset
--drop-original      remove the source audio column
```

The audio column is found automatically by looking for a `datasets.Audio`
feature, preferring conventional names (`audio`, `speech`, `wav`, …). Override
with `--audio-column` when a dataset has more than one.

### Diffusion steps

**50 is the default and the recommendation.** In listening tests on this
checkpoint, 25 — the Seed-VC default — was noticeably robotic, and 50 fixed it.
100 is available but the further gain is small.

| Steps | Character |
| --- | --- |
| 25 | Fastest, audibly robotic |
| **50** | **Recommended** — smooth, good speed |
| 100 | Marginally smoother, ~2x slower than 50 |

```bash
twi-zephyr-vc convert --dataset org/ds --output org/ds-zephyr --diffusion-steps 100
```

### Local files and folders

No Hub dataset needed — point it at files or directories on disk:

```bash
# a folder (recurses by default), writing converted audio to out/
twi-zephyr-vc convert-local recordings/ -o out/

# several inputs at once, mixing files and folders
twi-zephyr-vc convert-local clip.wav interviews/ more/*.mp3 -o out/

# convert locally, then publish the result to the Hub as well
twi-zephyr-vc convert-local recordings/ -o out/ --push-to my-org/my-zephyr-audio
```

Reads `.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.aac`, `.wma` and
writes `.wav`. Directory structure under each input folder is preserved in the
output. Files already present in the output are skipped unless you pass
`--overwrite`, so an interrupted run can simply be restarted.

```
inputs               files and/or directories             [required]
-o, --output-dir     where to write converted audio       [required]
--no-recursive       don't descend into subdirectories
--overwrite          reconvert files already in the output
--push-to            also publish results to the Hub (org/name)
--private            push privately
--diffusion-steps    25 | 50 | 100        (default: 50)
```

### A single file

```bash
twi-zephyr-vc convert-file input.wav -o output.wav --diffusion-steps 50
```

### From Python

```python
from twi_zephyr_vc import ZephyrConverter, convert_dataset, convert_paths

# one clip
conv = ZephyrConverter(diffusion_steps=50)
wav, sr = conv.convert_file("input.wav")

# local files or folders
convert_paths(["recordings/"], "out/", diffusion_steps=50)

# a whole dataset
convert_dataset(
    "org/source-dataset",
    "org/output-dataset",
    num_samples=100,
    diffusion_steps=50,
)
```

`ZephyrConverter` loads the checkpoint once and reuses it, so converting many
clips in a loop does not reload the model each time.

## Notes

Conversion runs one clip at a time — Seed-VC's inference path is not batched.
Budget roughly a second or two per clip of speech on a modern GPU at 50 steps,
more at 100.

If a clip fails to convert, the run continues: that row keeps its original audio
and a warning is logged with the index, so a single bad file will not kill a long
job. The count of failures is reported at the end.

## Licensing

This library is **GPL-3.0-or-later**. It imports Seed-VC at runtime, and Seed-VC
is GPL-3.0, so a compatible copyleft licence is the honest choice here.

The **model** is a separate matter and carries its own restrictions — most
importantly its training data is CC-BY-NC-4.0, making model use
**non-commercial**. Read the
[model card](https://huggingface.co/ghanaopenai/twi-zephyr-vc) before using this
for anything beyond research.

## Credits

- [Seed-VC](https://github.com/Plachtaa/seed-vc) by Plachtaa — architecture and
  pretrained weights
- Training data: [Ghana NLP Community](https://huggingface.co/ghananlpcommunity)
