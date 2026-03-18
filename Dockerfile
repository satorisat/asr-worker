# RunPod base image — PyTorch 2.2.0 + CUDA 12.1 + Python 3.10
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Fix torchvision/transformers version conflicts
RUN pip install --no-cache-dir \
    torchvision==0.17.0 \
    transformers==4.39.3 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY handler.py .

CMD ["python", "-u", "handler.py"]
