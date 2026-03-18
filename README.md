# ASR Worker
Serverless worker for audio transcription.
Dual-model: GigaAM v3 (Russian) + WhisperX large-v3 (other languages).

## Providers
- **Modal** — `modal_worker.py` (primary)
- **RunPod** — `handler.py` + `Dockerfile`

## Deploy
Push to main → GitHub Actions автоматически деплоит в Modal.
