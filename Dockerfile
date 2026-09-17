# Prebuilt ghana-vc runtime.
#
# The point of this image is that dependency resolution happens here, once, at
# build time -- not on the user's GPU at runtime. Seed-VC's requirements.txt
# fights whatever is already installed (it has clobbered both torch/torchaudio
# and huggingface_hub in practice), and the outcome varies with base image and
# install order. Resolving it once and shipping the result removes that whole
# class of failure, and saves ~4-5 minutes of pip per job.
#
# Build:
#   docker build -t ghcr.io/ghanaopenai/ghana-vc:latest .
# Run:
#   docker run --gpus all -e HF_TOKEN=... ghcr.io/ghanaopenai/ghana-vc:latest \
#     ghana-vc convert --dataset org/ds --output org/out --num-samples 10

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GHANA_VC_HOME=/opt/ghana-vc \
    HF_HOME=/opt/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        build-essential gcc g++ \
        git ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip

# 1. Seed-VC first, so its requirements own the torch stack uncontested.
ARG SEEDVC_REF=51383efd921027683c89e5348211d93ff12ac2a8
RUN git clone https://github.com/Plachtaa/seed-vc.git ${GHANA_VC_HOME}/seed-vc \
    && cd ${GHANA_VC_HOME}/seed-vc \
    && git checkout ${SEEDVC_REF} \
    && python -m pip install -r requirements.txt

# 2. ghana-vc on top. Its own deps are light and deliberately exclude torch.
ARG GHANA_VC_REF=main
RUN python -m pip install "git+https://github.com/GhanaNLP/ghana-vc@${GHANA_VC_REF}"

# 3. Do NOT upgrade huggingface_hub: Seed-VC's BigVGAN calls the older Hub API
#    and fails with "BigVGAN._from_pretrained() missing 2 required keyword-only
#    arguments" if it moves. datasets is held on 2.x instead, and only-if-needed
#    stops pip dragging numpy past Seed-VC's numpy==1.26.4.
RUN python -m pip install --upgrade-strategy only-if-needed \
        "datasets>=2.18,<3" "huggingface_hub>=0.28.1,<0.34"

# 4. Bake the weights in so the first conversion does not wait on a 412 MB
#    download. Public repos, so no token is needed at build time.
RUN python -c "\
from huggingface_hub import hf_hub_download, snapshot_download; \
[hf_hub_download('ghanaopenai/ghana-vc', f) for f in \
 ('ft_model.pth','config_dit_mel_seed_uvit_whisper_small_wavenet.yml','ref_zephyr.wav')]; \
snapshot_download('openai/whisper-small', allow_patterns=['*.json','*.txt','*.safetensors'])"

# 5. Fail the build if the stack is broken, rather than shipping it.
RUN python -c "\
import torch, torchaudio, datasets, huggingface_hub, soundfile, numpy; \
import ghana_vc; \
from ghana_vc import ZephyrConverter; \
print('torch', torch.__version__, '| torchaudio', torchaudio.__version__); \
print('datasets', datasets.__version__, '| hub', huggingface_hub.__version__); \
print('numpy', numpy.__version__); \
print('ghana_vc', ghana_vc.__version__)"

WORKDIR /work
CMD ["ghana-vc", "--help"]
