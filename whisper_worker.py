"""
Modal Serverless Worker — Multilingual Transcription
Whisper large-v3-turbo (all languages except Russian) + pyannote diarization

Deploy:
  modal deploy whisper_worker.py
"""

import os
import tempfile
import subprocess
import traceback

import modal

app = modal.App("whisper-worker")

volume = modal.Volume.from_name("asr-models-cache", create_if_missing=True)
CACHE_DIR = "/vol/hf_cache"

MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


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


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _guess_suffix(url: str) -> str:
    path = url.split("?")[0]
    ext = os.path.splitext(path)[1]
    return ext if ext else ".audio"


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "faster-whisper==1.1.1",
        "requests",
        "huggingface_hub>=0.20.0,<1.0",
        "fastapi[standard]",
        "soundfile",
        "matplotlib",
        "scipy",
        "sentencepiece",
        "pyannote.audio==3.3.2",
        "speechbrain>=1.0.0",
        "scikit-learn>=1.3.0",
    )
)


# ---------------------------------------------------------------------------
# Worker class
# ---------------------------------------------------------------------------

@app.cls(
    gpu="L4",
    image=image,
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("worker-auth-token"),
    ],
    volumes={CACHE_DIR: volume},
    scaledown_window=2,
    timeout=3600,
    enable_memory_snapshot=True,
)
class WhisperWorker:

    @modal.enter()
    def load_models(self):
        import torch
        from faster_whisper import WhisperModel
        from pyannote.audio import Pipeline

        self._load_error = None

        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = os.environ.get("HF_TOKEN", "")

        print(f"Device: {self.device}")

        print("Loading Whisper large-v3-turbo...")
        try:
            self.whisper_model = WhisperModel(
                "large-v3-turbo",
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8",
                download_root=os.path.join(CACHE_DIR, "whisper"),
            )
            print("Whisper loaded.")
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
            print(f"FATAL: Whisper failed to load:\n{traceback.format_exc()}")
            return  # skip diarization loading too

        print("Loading pyannote diarization...")
        self.diarize_model = None
        if hf_token:
            try:
                from contextlib import nullcontext
                from torch.torch_version import TorchVersion
                from pyannote.audio.core.task import Problem, Resolution, Specifications
                os.environ["HF_HUB_OFFLINE"] = "0"
                try:
                    if hasattr(torch.serialization, "safe_globals"):
                        ctx = torch.serialization.safe_globals([TorchVersion, Problem, Specifications, Resolution])
                    else:
                        ctx = nullcontext()
                    with ctx:
                        self.diarize_model = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=hf_token,
                        ).to(torch.device(self.device))
                    print("Diarization model loaded.")
                finally:
                    os.environ["HF_HUB_OFFLINE"] = "1"
            except Exception as e:
                print(f"Warning: diarization failed to load: {e}")

        print("All models loaded. Worker ready.")

    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, request: dict) -> dict:
        if getattr(self, "_load_error", None):
            return {"error": f"Worker initialization failed: {self._load_error}"}

        audio_url = request.get("audio_url")
        language = request.get("language")  # None = auto-detect
        enable_diarization = request.get("enable_diarization", True)
        min_speakers = _safe_int(request.get("min_speakers", 1), 1)
        max_speakers = _safe_int(request.get("max_speakers", 10), 10)

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
        wav_path = None
        try:
            suffix = _guess_suffix(audio_url)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(tmp_fd)

            # Скачиваем аудио
            dl = subprocess.run(
                ["ffmpeg", "-y", "-i", audio_url, "-c", "copy", tmp_path],
                capture_output=True,
            )
            if dl.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1024:
                return {"error": f"Download failed: {dl.stderr.decode()[:300]}"}

            if os.path.getsize(tmp_path) > MAX_DOWNLOAD_BYTES:
                return {"error": f"Audio file exceeds {MAX_DOWNLOAD_BYTES // (1024 ** 3)} GB limit"}

            # Конвертируем в WAV 16kHz (faster-whisper требует)
            wav_path = tmp_path + ".wav"
            conv = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True,
            )
            if conv.returncode != 0 or not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1024:
                raise RuntimeError(f"WAV conversion failed: {conv.stderr.decode()[:300]}")

            return self._run_whisper(
                wav_path, language, enable_diarization, min_speakers, max_speakers
            )

        except Exception as e:
            print(traceback.format_exc())
            return {"error": str(e)}
        finally:
            for p in (tmp_path, wav_path):
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception as exc:
                        print(f"Warning: failed to delete temp file {p}: {exc}")

    def _run_whisper(self, audio_path, language, enable_diarization, min_speakers, max_speakers):
        import concurrent.futures

        print(f"Running Whisper large-v3-turbo (language={language or 'auto'})...")

        lang_arg = language if language and language != "auto" else None

        if enable_diarization and self.diarize_model:
            # Запускаем транскрипцию и диаризацию параллельно
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                whisper_future = executor.submit(self._transcribe, audio_path, lang_arg)
                diarize_future = executor.submit(self._diarize, audio_path, min_speakers, max_speakers)

                segments, detected_language, duration = whisper_future.result()
                diarize_result = diarize_future.result()

            if diarize_result is not None and segments:
                segments = self._merge_speakers(segments, diarize_result)
            else:
                for seg in segments:
                    seg["speaker"] = "SPEAKER_00"
        else:
            segments, detected_language, duration = self._transcribe(audio_path, lang_arg)
            if enable_diarization:
                # Diarization was requested but model not available
                for seg in segments:
                    seg["speaker"] = "SPEAKER_00"

        return self._build_response(segments, detected_language, duration)

    def _transcribe(self, audio_path, lang_arg):
        segments_iter, info = self.whisper_model.transcribe(
            audio_path,
            language=lang_arg,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        segments = []
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                segments.append({
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": text,
                })

        detected_language = info.language or "unknown"
        duration = float(info.duration) if info.duration else (
            segments[-1]["end"] if segments else 0.0
        )
        print(f"Whisper done: {len(segments)} segments, language={detected_language}")
        return segments, detected_language, duration

    def _diarize(self, audio_path, min_speakers, max_speakers):
        print("Running diarization...")
        try:
            result = self.diarize_model(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            print("Diarization done.")
            return result
        except Exception as e:
            print(f"Diarization failed ({type(e).__name__}): {e}")
            return None

    def _merge_speakers(self, segments, diarize_result):
        speaker_turns = []
        try:
            for turn, _, speaker in diarize_result.itertracks(yield_label=True):
                speaker_turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        except Exception as e:
            print(f"Speaker merge failed (itertracks): {e} — assigning SPEAKER_00 to all segments")
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"
            return segments

        for seg in segments:
            seg_start, seg_end = seg["start"], seg["end"]
            best_speaker, best_overlap = "SPEAKER_00", 0.0
            for turn in speaker_turns:
                overlap = min(seg_end, turn["end"]) - max(seg_start, turn["start"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn["speaker"]
            seg["speaker"] = best_speaker

        # Перенумеровываем по порядку первого появления
        speaker_order = {}
        for seg in segments:
            sp = seg.get("speaker")
            if sp and sp not in speaker_order:
                speaker_order[sp] = len(speaker_order)
        for seg in segments:
            sp = seg.get("speaker")
            if sp in speaker_order:
                seg["speaker"] = f"SPEAKER_{speaker_order[sp]:02d}"

        print(f"Speakers merged: {len(speaker_order)} speakers")
        return segments

    def _build_response(self, segments, language, duration):
        raw_text = " ".join(s["text"] for s in segments)
        word_count = len(raw_text.split()) if raw_text else 0

        has_speakers = any(seg.get("speaker") for seg in segments)
        formatted_text = self._build_formatted_text(segments) if has_speakers else raw_text

        return {
            "text": raw_text,
            "formatted_text": formatted_text,
            "segments": segments,
            "language": language,
            "duration": duration,
            "word_count": word_count,
            "diarization_available": self.diarize_model is not None,
        }

    def _build_formatted_text(self, segments):
        lines, current_speaker, current_texts = [], None, []
        for seg in segments:
            speaker = seg.get("speaker") or "SPEAKER_00"
            text = seg.get("text", "").strip()
            if not text:
                continue
            if speaker != current_speaker:
                if current_speaker and current_texts:
                    lines.append(f"**{self._fmt_speaker(current_speaker)}:** {' '.join(current_texts)}")
                current_speaker, current_texts = speaker, [text]
            else:
                current_texts.append(text)
        if current_speaker and current_texts:
            lines.append(f"**{self._fmt_speaker(current_speaker)}:** {' '.join(current_texts)}")
        return "\n\n".join(lines)

    def _fmt_speaker(self, speaker_id):
        if speaker_id.startswith("SPEAKER_"):
            try:
                idx = int(speaker_id.replace("SPEAKER_", ""))
                return f"Спикер {idx + 1}"
            except ValueError:
                pass
        return speaker_id
