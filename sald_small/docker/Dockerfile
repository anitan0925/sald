FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /home

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Asia/Tokyo \
    apt-get install -y \
    git \
    emacs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python - <<'PY'
import torch
print("torch:", torch.__version__)
print("has torch.xpu:", hasattr(torch, "xpu"))

import huggingface_hub
print("huggingface_hub:", huggingface_hub.__version__)

from diffusers import DDPMPipeline, DDPMScheduler
print("diffusers import: OK")
PY

CMD ["/bin/bash"]
