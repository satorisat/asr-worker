"""
Modal Serverless Worker — многоязычная транскрибация (казахский, киргизский, узбекский)
GigaAM multilingual_large_ctc + pyannote speaker diarization

Копия modal_worker.py с одной заменой: модель `multilingual_large_ctc` вместо
`v3_e2e_rnnt`. У этой модели CTC-голова и посимвольный словарь, языкового параметра
нет — один и тот же код пишет казахский и киргизский кириллицей, узбекский латиницей.
Какой язык отправлять в этот воркер, решает бэкенд; поле `language` из запроса
возвращается в ответе как есть.

ВЕСА КЛАДУТСЯ В ТОМ ЗАРАНЕЕ, воркер их НЕ качает. Причина: библиотека берёт веса с
CDN Сбера, а у него с 04.11.2026 просрочен сертификат (проверено `openssl s_client`:
Let's Encrypt YR1, 06.08.2026–04.11.2026, автопродление не сработало). Зеркало
`ai-sage/GigaAM-Multilingual@large_ctc` на HuggingFace не подходит: там плоский словарь
тензоров (`model.encoder.*`) без ключа `cfg`, а библиотека строит модель из `cfg`
(проверено скачиванием: md5 c77570645840c876f173ac8e14e71ef9 против нужного
79a9adde50dd7f35bbf70927cb6557d0, те же 2233 МБ весов, другая упаковка).
Разовая укладка весов — `provision_ml_weights.py`, там же сверка MD5.
Подмену закрывает сама библиотека: `load_model` делает
`assert hash_path(path) == _MODEL_HASHES[name]` при КАЖДОЙ загрузке, то есть битый или
подложенный файл роняет контейнер, а не уезжает в работу молча.

Deploy:
  pip install modal
  modal token new
  modal deploy modal_worker_ml.py
"""

import json
import os
import tempfile
import subprocess
import traceback

import modal

app = modal.App("asr-worker-ml")

# Отдельный том, а не общий asr-models-cache: прод-воркеры (русский и whisper) его не
# монтируют, значит укладка 2,2 ГБ весов сюда не может задеть работающие воркеры.
volume = modal.Volume.from_name("asr-models-cache-ml", create_if_missing=True)
CACHE_DIR = "/vol/hf_cache"
MODEL_NAME = "multilingual_large_ctc"

os.environ.setdefault("HF_XET_CACHE", "/tmp/hf_xet")

MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024


def _validate_url(url: str) -> None:
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
        return
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        raise ValueError(f"Blocked private address: {host}")


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hf_offline(on: bool):
    v = "1" if on else "0"
    os.environ["HF_HUB_OFFLINE"] = v
    os.environ["TRANSFORMERS_OFFLINE"] = v


def _load_offline_first(loader, what: str):
    _hf_offline(True)
    try:
        return loader()
    except Exception as e:
        print(f"{what}: offline cache miss ({type(e).__name__}: {e}) — downloading once online")
        _hf_offline(False)
        try:
            return loader()
        finally:
            _hf_offline(True)


# Образ — копия продового, с теми же пинами (урезанный образ = урезанный тест).
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
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
        "pyannote.audio==3.3.2",
        "speechbrain>=1.0.0",
        "scikit-learn>=1.3.0",
        "git+https://github.com/salute-developers/GigaAM.git",
    )
    .env({
        "HF_HOME": CACHE_DIR,
        "PYANNOTE_CACHE": CACHE_DIR + "/pyannote",
        "HF_XET_CACHE": "/tmp/hf_xet",
    })
)


# ---------------------------------------------------------------------------
# Воркер — копия продового класса, другая модель
# ---------------------------------------------------------------------------

@app.cls(
    # Тот же список карт и тот же порядок, что у русского воркера: Modal берёт первую
    # СВОБОДНУЮ и порядок соблюдает. Обоснование — в modal_worker.py.
    gpu=["L4", "A10", "L40S", "A100", "H100"],
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
class ASRWorkerML:

    @modal.enter(snap=True)
    def load_models(self):
        import shutil
        import time
        import torch
        from pyannote.audio import Pipeline
        import gigaam

        self._load_error = None
        self.device = "cpu"

        os.environ["HF_HOME"] = CACHE_DIR
        _hf_offline(True)
        torch.hub.set_dir(CACHE_DIR)

        hf_token = os.environ.get("HF_TOKEN", "")

        import gigaam.vad_utils as _vad
        def _patched_load_segmentation_model(model_id):
            from pyannote.audio import Model
            from torch.torch_version import TorchVersion
            from pyannote.audio.core.task import Problem, Resolution, Specifications
            def _load():
                with torch.serialization.safe_globals([TorchVersion, Problem, Specifications, Resolution]):
                    return Model.from_pretrained(model_id, use_auth_token=os.environ.get("HF_TOKEN"))
            return _load_offline_first(_load, f"VAD segmentation ({model_id})")
        _vad.load_segmentation_model = _patched_load_segmentation_model

        gigaam_cache = os.path.join(CACHE_DIR, "gigaam")
        os.makedirs(gigaam_cache, exist_ok=True)

        print(f"Loading GigaAM {MODEL_NAME} (CPU, snapshot phase)...")
        last_err = None
        downloaded_fresh = False
        for attempt in range(1, 6):
            try:
                had_files_before = bool(os.listdir(gigaam_cache))
                self.gigaam_model = gigaam.load_model(MODEL_NAME, download_root=gigaam_cache)
                if not had_files_before:
                    downloaded_fresh = True
                break
            except Exception as e:
                last_err = e
                print(f"GigaAM load attempt {attempt}/5 failed: {type(e).__name__}: {e}")
                shutil.rmtree(gigaam_cache, ignore_errors=True)
                os.makedirs(gigaam_cache, exist_ok=True)
                if attempt < 5:
                    time.sleep(min(2 ** attempt, 30))
        else:
            print(f"FATAL: GigaAM failed to load after 5 attempts:\n{traceback.format_exc()}")
            raise RuntimeError(f"GigaAM load failed: {type(last_err).__name__}: {last_err}")
        print("GigaAM loaded (CPU)")

        if downloaded_fresh:
            try:
                volume.commit()
                print("GigaAM weights committed to volume")
            except Exception as e:
                print(f"Warning: volume.commit() failed: {e}")

        print("Loading pyannote diarization (CPU, offline-first)...")
        self.diarize_model = None
        if hf_token:
            try:
                from torch.torch_version import TorchVersion
                from pyannote.audio.core.task import Problem, Resolution, Specifications

                def _load_diarize():
                    with torch.serialization.safe_globals([TorchVersion, Problem, Specifications, Resolution]):
                        return Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=hf_token,
                        )

                self.diarize_model = _load_offline_first(_load_diarize, "pyannote diarization")
                print("Diarization loaded (CPU)")
            except Exception as e:
                print(f"Warning: diarization failed to load: {e}")

        print("Snapshot-phase load complete (models on CPU).")

    @modal.enter(snap=False)
    def move_to_gpu(self):
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        if self.device != "cuda":
            print("No GPU available — models stay on CPU.")
            return

        if getattr(self, "gigaam_model", None) is not None and hasattr(self.gigaam_model, "to"):
            self.gigaam_model = self.gigaam_model.to(self.device)
        if getattr(self, "diarize_model", None) is not None:
            try:
                self.diarize_model = self.diarize_model.to(torch.device(self.device))
            except Exception as e:
                print(f"Warning: moving diarization to GPU failed ({e}) — disabling diarization")
                self.diarize_model = None
        print(f"Models moved to {self.device}. Worker ready.")

    @modal.method()
    def do_transcribe(self, request: dict) -> dict:
        if getattr(self, "_load_error", None):
            return {"error": f"Worker initialization failed: {self._load_error}"}

        audio_url = request.get("audio_url")
        language = request.get("language")
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
                "language": language or "mul",
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
        result = self.gigaam_model.transcribe_longform(input_path)
        utterances = getattr(result, "segments", result)
        segments = []
        for utt in utterances:
            if isinstance(utt, dict):
                text = (utt.get("transcription") or "").strip()
                start, end = utt["boundaries"]
            else:
                text = (getattr(utt, "text", "") or "").strip()
                start, end = utt.start, utt.end
            if text:
                segments.append({"start": float(start), "end": float(end), "text": str(text)})
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
# FastAPI app — submit / poll pattern (как в modal_worker.py)
# ---------------------------------------------------------------------------

# Отдельный лёгкий образ для HTTP-слоя: он только принимает задание, вызывает spawn и
# отдаёт результат. Обоснование и пин версии — в modal_worker.py, здесь то же самое.
web_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "fastapi[standard]==0.143.0"
)


@app.function(
    image=web_image,
    # 300с, а не 2: бэкенд опрашивает результат раз в 5 секунд, при двух секундах
    # контейнер успевал умереть между опросами. Подробности — в modal_worker.py.
    scaledown_window=300,
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
            worker = ASRWorkerML()
            fc = await worker.do_transcribe.spawn.aio(request)
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
