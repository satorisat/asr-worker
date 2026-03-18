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

# Pre-download whisperx and gigaam models (no auth needed)
RUN python -c "\
import whisperx; \
whisperx.load_model('tiny', 'cpu', compute_type='float32'); \
whisperx.load_model('large-v3', 'cpu', compute_type='float32'); \
"

RUN python -c "\
import gigaam; \
gigaam.load_model('v3_e2e_rnnt'); \
"

# Note: pyannote/speaker-diarization-3.1 requires HF_TOKEN and is downloaded
# on first cold start using the HF_TOKEN env var set in RunPod endpoint settings.

# Copy worker code
COPY handler.py .

CMD ["python", "-u", "handler.py"]
