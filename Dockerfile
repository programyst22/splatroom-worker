FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

USER root
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SPLATROOM_WORKDIR=/tmp/splatroom \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"

# Runtime + reconstruction tools.
# Ubuntu's COLMAP package is sufficient for the first production test;
# we can switch to a CUDA-built COLMAP later for faster preprocessing.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      colmap \
      git \
      ca-certificates \
      curl \
      build-essential \
      cmake \
      ninja-build \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Keep the Blackwell-capable PyTorch already present in the base image.
RUN python -m pip install --no-cache-dir --upgrade pip wheel "setuptools<82"

# Nerfstudio 1.1.5 uses gsplat 1.4.0. PyTorch 2.8 + CUDA 12.8
# supports Blackwell (sm_120), unlike the old CUDA 11.8 image.
RUN python -m pip install --no-cache-dir "nerfstudio==1.1.5"

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY preflight.py /app/preflight.py

# Fail the image build if the expected CLIs/modules are missing and
# verify that pip did not replace our CUDA 12.8 PyTorch build.
RUN python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
assert torch.__version__.startswith("2.8."), torch.__version__
assert str(torch.version.cuda).startswith("12.8"), torch.version.cuda
PY

RUN python /app/preflight.py --build-check

CMD ["python", "-u", "/app/handler.py"]
