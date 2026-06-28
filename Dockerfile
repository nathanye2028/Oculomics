# Reproducible CUDA environment for training on a remote GPU server.
#
#   docker build -t oculomics .
#   docker run --gpus all --rm -it -v $PWD:/workspace/Oculomics oculomics \
#       python3 run_experiment.py --seeds 0 1 2 --epochs 200 --patch-size 512 --eval-tiled --amp
#
# `python3 check_env.py` (the default CMD) verifies the GPU is visible first.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Oculomics

# Install CUDA-built torch first (cu121), then the rest. requirements pins the
# same torch==2.8.0 version, so pip treats it as already-satisfied (no CPU
# rebuild / downgrade).
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel \
    && python3 -m pip install torch==2.8.0 torchvision==0.23.0 \
         --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -m pip install -r requirements.txt

COPY . .

CMD ["python3", "check_env.py"]
