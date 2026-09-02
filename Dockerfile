# Reproducible CUDA environment for training on a remote GPU server.
#
#   docker build -t oculomics .
#   docker run --gpus all --rm -it -v $PWD:/workspace/Oculomics oculomics \
#       python3 run_experiment.py --seeds 0 1 2 --epochs 200 --patch-size 512 --eval-tiled --amp
#
# `python3 check_env.py` (the default CMD) verifies the GPU is visible first.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Oculomics

# Install CUDA-built torch first, then the rest. requirements pins the same
# torch==2.8.0 version, so pip treats it as already-satisfied (no CPU rebuild /
# downgrade). NB: torch 2.8.0 is published for cu126/cu128 only — the cu121
# index does not carry it and the build fails there.
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel \
    && python3 -m pip install torch==2.8.0 torchvision==0.23.0 \
         --index-url https://download.pytorch.org/whl/cu126 \
    && python3 -m pip install -r requirements.txt

# .dockerignore mirrors .gitignore: data/ (8 GB, FGADR is non-redistributable),
# checkpoints, exports and logs never enter the build context.
COPY . .

CMD ["python3", "check_env.py"]
