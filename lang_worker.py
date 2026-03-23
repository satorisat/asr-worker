"""
Modal Serverless Worker — Language Detection
Whisper tiny (CPU) — определяет язык по первым 30 сек аудио

Deploy:
  modal deploy lang_worker.py
"""

import os
import tempfile
import subprocess
import traceback

import modal

app = modal.App("lang-worker")

volume = modal.Volume.from_name("asr-models-cache", create_if_missing=True)
CACHE_DIR = "/vol/hf_cache"


def _validate_url(url: str) -> None:
    """Raise ValueError if URL is unsafe (SSRF prevention)."""
    import urllib.parse
    import ipaddress
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")
    host = parsed.hostname or ""
    if not host:
        raise ValueError("URL missing host")
    if host in ("localhost", "::1"):
        raise ValueError(f"Blocked host: {host}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return  # domain name — allowed
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        raise ValueError(f"Blocked private address: {host}")


def _guess_suffix(url: str) -> str:
    """Вытаскиваем расширение из URL, fallback → .audio"""
    path = url.split("?")[0]
    ext = os.path.splitext(path)[1]
    return ext if ext else ".audio"


# ---------------------------------------------------------------------------
# Image — faster-whisper (CPU)
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper==1.1.1",
        "huggingface_hub>=0.20.0,<1.0",
        "fastapi[standard]",
    )
)


# ---------------------------------------------------------------------------
# Worker class
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("worker-auth-token", required=False)],
    volumes={CACHE_DIR: volume},
    scaledown_window=2,
    timeout=120,
    enable_memory_snapshot=True,
)
class LangWorker:

    @modal.enter()
    def load_models(self):
        from faster_whisper import WhisperModel

        self._load_error = None

        os.environ["HF_HOME"] = CACHE_DIR

        print("Loading Whisper tiny...")
        try:
            # cpu + int8 — минимальные ресурсы, достаточная точность для языка
            self.model = WhisperModel(
                "tiny",
                device="cpu",
                compute_type="int8",
                download_root=os.path.join(CACHE_DIR, "whisper"),
            )
            print("Whisper tiny loaded. Worker ready.")
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
            print(f"FATAL: Whisper tiny failed to load:\n{traceback.format_exc()}")

    @modal.fastapi_endpoint(method="POST")
    def detect(self, request: dict) -> dict:
        if getattr(self, "_load_error", None):
            return {"error": f"Worker initialization failed: {self._load_error}"}

        audio_url = request.get("audio_url")

        worker_token = os.environ.get("WORKER_AUTH_TOKEN", "")
        if worker_token and request.get("auth_token") != worker_token:
            return {"error": "Unauthorized"}

        if not audio_url:
            return {"error": "audio_url is required"}

        try:
            _validate_url(audio_url)
        except ValueError as e:
            return {"error": str(e)}

        tmp_path = None
        try:
            # Скачиваем и конвертируем первые 30 сек в WAV 16kHz
            suffix = _guess_suffix(audio_url)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(tmp_fd)

            dl = subprocess.run(
                ["ffmpeg", "-y", "-i", audio_url,
                 "-t", "30",          # только первые 30 секунд
                 "-ar", "16000",      # 16kHz — требование Whisper
                 "-ac", "1",          # моно
                 "-f", "wav", tmp_path],
                capture_output=True,
            )
            if dl.returncode != 0:
                return {"error": f"Download/convert failed: {dl.stderr.decode()[:300]}"}

            file_size = os.path.getsize(tmp_path)
            if file_size < 1024:
                return {"error": "Audio too short or empty"}

            # Определяем язык
            _, info = self.model.transcribe(
                tmp_path,
                task="transcribe",
                language=None,        # auto-detect
                beam_size=1,          # быстрее, для языка достаточно
                without_timestamps=True,
            )

            return {
                "language": info.language,
                "probability": round(info.language_probability, 3),
            }

        except Exception as e:
            return {"error": str(e)}
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as exc:
                    print(f"Warning: failed to delete temp file {tmp_path}: {exc}")
