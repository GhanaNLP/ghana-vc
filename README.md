# ghana-vc

Voice conversion for Ghanaian languages. Convert **any** speech — a local file,
a folder, or a whole Hugging Face dataset — into a single consistent Ghanaian
voice, and push the result back to the Hub.

Built on [Seed-VC](https://github.com/Plachtaa/seed-vc) with the
[`ghanaopenai/ghana-vc`](https://huggingface.co/ghanaopenai/ghana-vc)
checkpoint.

## About the voice, and how far it travels

The checkpoint was **fine-tuned on a Twi (Akan) voice** — the target speaker,
*Zephyr*, is a Twi speaker, and Twi is what the model saw during training.

In practice it **transfers well to other languages**. Voice conversion operates
on speaker timbre rather than on words, so the model is not bound to the
language it was trained on. We have converted speech across 40+ Ghanaian and
West African languages — Ewe, Dagbani, Ga, Dangme, Gonja, Kusaal, Hausa and
many more — and the target voice comes through clearly.

**Hear it for yourself:** [ghana-vc demo](https://huggingface.co/spaces/ghanaopenai/ghana-vc-demo)
— three samples from each of 40+ languages, original beside converted.

Two honest caveats:

- Quality is **best on Twi** and merges gradually as a language's phonology
  moves further from Akan. Languages with sounds absent from the training data
  can show occasional artifacts.
- The checkpoint is **not the ceiling**. It was fine-tuned for 2,500 steps on
  Twi alone. Fine-tuning further — on more Twi, or on a target voice in the
  language you care about — should improve things, and the same
  [Seed-VC](https://github.com/Plachtaa/seed-vc) recipe applies.

If you fine-tune a better checkpoint, point the library at it with
`--model-repo <org>/<name>`; nothing else has to change.

## Install

```bash
pip install git+https://github.com/GhanaNLP/ghana-vc
```

Not yet on PyPI — install from source as above.

Seed-VC itself is not on PyPI, so it is cloned into `~/.cache/ghana-vc/`
on first use and its requirements installed. Set `GHANA_VC_HOME` to change
that location, or pass `--no-install-deps` if you manage the environment
yourself.

`torch`, `torchaudio` and `librosa` are intentionally **not** pinned by this
package — Seed-VC's own `requirements.txt` owns them. Installing a second torch
build alongside Seed-VC's leads to a CUDA runtime mismatch
(`libcudart.so.12: cannot open shared object file`). Install into a clean
environment and let Seed-VC pull the torch stack it expects.

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

## Run it on a GPU service

A prebuilt image is published so you don't resolve dependencies on a GPU:

```
ghcr.io/ghanaopenai/ghana-vc:latest
```

Seed-VC, the torch stack and the checkpoint are all baked in, so a container
starts converting in seconds instead of spending ~5 minutes on `pip install` —
billed GPU minutes on most services.

### Hugging Face Jobs

```bash
hf jobs run --flavor l4x1 --timeout 2h -s HF_TOKEN \
  ghcr.io/ghanaopenai/ghana-vc:latest \
  ghana-vc convert --dataset <org>/<ds> --output <org>/<out> --num-samples 100
```

Mount a local folder with `-v ./audio:/work/audio` and use `convert-local`
instead, to convert files from disk.

### Modal

```bash
modal run examples/modal_app.py --dataset <org>/<ds> --output <org>/<out>
```

See [`examples/modal_app.py`](examples/modal_app.py). It expects a Modal secret
named `huggingface` providing `HF_TOKEN`.

### Docker, anywhere

```bash
docker run --gpus all -e HF_TOKEN=$HF_TOKEN \
  -v "$PWD/audio:/work/audio" -v "$PWD/out:/work/out" \
  ghcr.io/ghanaopenai/ghana-vc:latest \
  ghana-vc convert-local audio/ -o out/
```

When using the image, pass `install_deps=False` from Python (the CLI detects
the preinstalled Seed-VC automatically via `GHANA_VC_HOME`).

## Use

### A Hugging Face dataset

Point it at a dataset, give it somewhere to push:

```bash
export HF_TOKEN=hf_...

ghana-vc convert \
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
ghana-vc convert --dataset org/ds --output org/ds-zephyr --diffusion-steps 100
```

### Local files and folders

No Hub dataset needed — point it at files or directories on disk:

```bash
# a folder (recurses by default), writing converted audio to out/
ghana-vc convert-local recordings/ -o out/

# several inputs at once, mixing files and folders
ghana-vc convert-local clip.wav interviews/ more/*.mp3 -o out/

# convert locally, then publish the result to the Hub as well
ghana-vc convert-local recordings/ -o out/ --push-to my-org/my-zephyr-audio
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
ghana-vc convert-file input.wav -o output.wav --diffusion-steps 50
```

### From Python

```python
from ghana_vc import ZephyrConverter, convert_dataset, convert_paths

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
[model card](https://huggingface.co/ghanaopenai/ghana-vc) before using this
for anything beyond research.

## Credits

- [Seed-VC](https://github.com/Plachtaa/seed-vc) by Plachtaa — architecture and
  pretrained weights
- Training data: [Ghana NLP Community](https://huggingface.co/ghananlpcommunity)
