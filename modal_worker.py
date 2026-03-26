"""
Modal Serverless Worker — Russian Audio Transcription
GigaAM v3 (Russian) + pyannote speaker diarization

Deploy:
  pip install modal
  modal token new
  modal deploy modal_worker.py
"""

import json
import os
import tempfile
import subprocess
import traceback

import modal

app = modal.App("asr-worker")

# Persistent volume for HF model cache — avoids re-downloading on every cold start
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


# ---------------------------------------------------------------------------
# Image — GigaAM + pyannote
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "git")
    # torch первым — чтобы torchcodec (dep pyannote 3.3) нашёл правильные CUDA-колёса
    .pip_install(
        "torch",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "requests",
        "huggingface_hub>=0.20.0,<1.0",
        "fastapi[standard]",
        "soundfile",
        "matplotlib",
        "scipy",
        "sentencepiece",
        "pyannote.audio==3.3.2",  # 3.3.2 = последний 3.x: numpy 2.x safe, без torchcodec
        "speechbrain>=1.0.0",
        "scikit-learn>=1.3.0",
        "git+https://github.com/salute-developers/GigaAM.git",
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
class ASRWorker:

    @modal.enter()
    def load_models(self):
        import torch
        from pyannote.audio import Pipeline
        import gigaam

        self._load_error = None

        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # GigaAM хранит веса в torch.hub dir — направляем в volume
        torch.hub.set_dir(CACHE_DIR)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = os.environ.get("HF_TOKEN", "")

        print(f"Device: {self.device}")

        # Patch GigaAM vad_utils: passes local snapshot path to Model.from_pretrained()
        # but pyannote 3.3.2 expects a HF repo ID. Override to pass repo ID directly.
        import gigaam.vad_utils as _vad
        def _patched_load_segmentation_model(model_id):
            from pyannote.audio import Model
            from torch.torch_version import TorchVersion
            from pyannote.audio.core.task import Problem, Resolution, Specifications
            with torch.serialization.safe_globals([TorchVersion, Problem, Specifications, Resolution]):
                return Model.from_pretrained(model_id, use_auth_token=os.environ.get("HF_TOKEN"))
        _vad.load_segmentation_model = _patched_load_segmentation_model

        print("Loading GigaAM v3...")
        try:
            self.gigaam_model = gigaam.load_model("v3_e2e_rnnt")
            if self.device == "cuda" and hasattr(self.gigaam_model, "to"):
                self.gigaam_model = self.gigaam_model.to(self.device)
                print(f"GigaAM moved to {self.device}")
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
            print(f"FATAL: GigaAM failed to load:\n{traceback.format_exc()}")
            return  # skip diarization loading too

        print("Loading pyannote diarization...")
        self.diarize_model = None
        if hf_token:
            try:
                from torch.torch_version import TorchVersion
                from pyannote.audio.core.task import Problem, Resolution, Specifications
                os.environ["HF_HUB_OFFLINE"] = "0"
                try:
                    with torch.serialization.safe_globals([TorchVersion, Problem, Specifications, Resolution]):
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

    @modal.method()
    def do_transcribe(self, request: dict) -> dict:
        if getattr(self, "_load_error", None):
            return {"error": f"Worker initialization failed: {self._load_error}"}

        audio_url = request.get("audio_url")
        enable_diarization = request.get("enable_diarization", True)
        min_speakers = _safe_int(request.get("min_speakers", 1), 1)
        max_speakers = _safe_int(request.get("max_speakers", 4), 4)

        if not audio_url:
            return {"error": "audio_url is required"}

        try:
            _validate_url(audio_url)
        except ValueError as e:
            return {"error": str(e)}

        audio_path = None
        try:
            print(f"Downloading: {audio_url[:80]}...")
            audio_path = self._download_audio(audio_url)
            duration = self._get_duration(audio_path)
            print(f"Duration: {duration:.1f}s")

            segments = self._run_gigaam(audio_path, enable_diarization, min_speakers, max_speakers)

            full_text = " ".join(s["text"] for s in segments if s.get("text"))
            formatted_text = self._build_formatted_text(segments) if enable_diarization else full_text

            return {
                "text": full_text,
                "formatted_text": formatted_text,
                "segments": segments,
                "language": "ru",
                "duration": duration,
                "word_count": len(full_text.split()),
                "diarization_available": self.diarize_model is not None,
            }

        except Exception as e:
            print(traceback.format_exc())
            return {"error": str(e)}

        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception as exc:
                    print(f"Warning: failed to delete temp file {audio_path}: {exc}")

    def _download_audio(self, url):
        import requests
        clean_url = url.split("?")[0]
        ext = clean_url.rsplit(".", 1)[-1].lower()
        if ext not in ("mp3", "wav", "ogg", "flac", "m4a", "mp4", "aac", "opus"):
            ext = "mp3"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
        tmp.close()
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = 0
            with open(tmp.name, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ValueError(f"Audio file exceeds {MAX_DOWNLOAD_BYTES // (1024 ** 3)} GB limit")
                    f.write(chunk)
        return tmp.name

    def _get_duration(self, audio_path):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
                capture_output=True, text=True,
            )
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
        except Exception as e:
            print(f"Warning: could not get audio duration: {e}")
            return 0.0

    def _run_gigaam(self, audio_path, enable_diarization, min_speakers, max_speakers):
        import concurrent.futures

        # Конвертируем один раз — используется и GigaAM, и pyannote
        wav_path = audio_path + ".wav"
        conv = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True,
        )
        if conv.returncode != 0 or not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1024:
            raise RuntimeError(f"ffmpeg WAV conversion failed: {conv.stderr.decode()[:300]}")

        try:
            if enable_diarization and self.diarize_model:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    transcribe_future = executor.submit(self._transcribe, wav_path)
                    diarize_future = executor.submit(self._diarize, wav_path, min_speakers, max_speakers)

                    segments = transcribe_future.result()
                    diarize_result = diarize_future.result()

                if diarize_result is not None and segments:
                    segments = self._assign_speakers(segments, diarize_result)
                else:
                    for seg in segments:
                        seg["speaker"] = "SPEAKER_00"
            else:
                segments = self._transcribe(wav_path)
                if enable_diarization:
                    for seg in segments:
                        seg["speaker"] = "SPEAKER_00"
        finally:
            try:
                os.unlink(wav_path)
            except Exception as exc:
                print(f"Warning: failed to delete wav file {wav_path}: {exc}")

        return segments

    def _transcribe(self, input_path):
        print("Running GigaAM transcribe_longform...")
        utterances = self.gigaam_model.transcribe_longform(input_path)
        segments = []
        for utt in utterances:
            text = utt["transcription"].strip()
            start, end = utt["boundaries"]
            if text:
                segments.append({"start": float(start), "end": float(end), "text": text})
        print(f"GigaAM transcribed {len(segments)} segments")
        return segments

    def _diarize(self, input_path, min_speakers, max_speakers):
        print("Running diarization...")
        try:
            kwargs = {}
            if min_speakers > 1:
                kwargs["min_speakers"] = min_speakers
            if max_speakers > 1:
                kwargs["max_speakers"] = max_speakers
            result = self.diarize_model(input_path, **kwargs)
            print("Diarization done.")
            return result
        except Exception as e:
            print(f"Diarization failed ({type(e).__name__}): {e}")
            return None

    def _assign_speakers(self, segments, diarize_result):
        speaker_turns = []
        try:
            for turn, _, speaker in diarize_result.itertracks(yield_label=True):
                speaker_turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        except Exception as e:
            print(f"Could not parse diarization result: {e} — assigning SPEAKER_00 to all segments")
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"
            return segments

        for seg in segments:
            seg_start, seg_end = seg.get("start", 0), seg.get("end", 0)
            best_speaker, best_overlap = "SPEAKER_00", 0.0
            for turn in speaker_turns:
                overlap = min(seg_end, turn["end"]) - max(seg_start, turn["start"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn["speaker"]
            seg["speaker"] = best_speaker

        # Renumber speakers by order of first appearance: SPEAKER_04 → SPEAKER_00, etc.
        speaker_order = {}
        for seg in segments:
            sp = seg.get("speaker")
            if sp and sp not in speaker_order:
                speaker_order[sp] = len(speaker_order)
        for seg in segments:
            sp = seg.get("speaker")
            if sp in speaker_order:
                seg["speaker"] = f"SPEAKER_{speaker_order[sp]:02d}"

        return segments

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


# ---------------------------------------------------------------------------
# FastAPI app — submit / poll pattern
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    scaledown_window=2,
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("worker-auth-token"),
    ],
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    fastapi_app = FastAPI()

    @fastapi_app.post("/transcribe")
    async def submit_transcribe(request: dict):
        worker_token = os.environ.get("WORKER_AUTH_TOKEN", "")
        if worker_token and request.get("auth_token") != worker_token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if not request.get("audio_url"):
            raise HTTPException(status_code=400, detail="audio_url is required")

        try:
            worker = ASRWorker()
            fc = worker.do_transcribe.spawn(request)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to spawn worker: {e}")

        return {"call_id": fc.object_id}

    @fastapi_app.get("/result/{call_id}")
    async def get_result(call_id: str):
        try:
            fc = modal.FunctionCall.from_id(call_id)
        except Exception as e:
            return JSONResponse(status_code=404, content={"error": f"Unknown call_id: {e}"})

        try:
            result = await fc.get.aio(timeout=0)
            return result
        except TimeoutError:
            return JSONResponse(status_code=202, content={"status": "processing"})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return fastapi_app
