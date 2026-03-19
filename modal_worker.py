"""
Modal Serverless Worker — Audio Transcription
Dual-model: GigaAM v3 (Russian) + WhisperX large-v3 (all other languages)
"""

import json
import os
import tempfile
import subprocess
import traceback

import modal

app = modal.App("asr-worker")


# ---------------------------------------------------------------------------
# Model pre-download function (runs during image build on CPU)
# ---------------------------------------------------------------------------

def download_models():
    from huggingface_hub import snapshot_download
    import gigaam
    import torch

    print("Downloading faster-whisper-tiny...")
    snapshot_download("Systran/faster-whisper-tiny")

    print("Downloading faster-whisper-large-v3...")
    snapshot_download("Systran/faster-whisper-large-v3")

    print("Downloading GigaAM v3...")
    gigaam.load_model("v3_e2e_rnnt")

    # Pre-download alignment models for common languages
    import whisperx
    print("Downloading alignment models...")
    for lang in ["en", "ru", "de", "fr", "es", "it", "zh", "ja", "ko", "pt"]:
        try:
            whisperx.load_align_model(language_code=lang, device="cpu")
            print(f"  Alignment model: {lang}")
        except Exception as e:
            print(f"  Alignment model {lang} skipped: {e}")

    print("All models downloaded.")


# ---------------------------------------------------------------------------
# Image — dependencies + pre-downloaded models baked in
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "requests",
        "huggingface_hub",
        "fastapi[standard]",
        "git+https://github.com/m-bain/whisperX.git",
        "git+https://github.com/salute-developers/GigaAM.git",
    )
    .run_function(download_models, cpu=2.0)
)


# ---------------------------------------------------------------------------
# Worker class
# ---------------------------------------------------------------------------

@app.cls(
    gpu="L4",
    image=image,
    secrets=[modal.Secret.from_name("hf-token")],
    scaledown_window=10,
)
class ASRWorker:

    @modal.enter()
    def load_models(self):
        import os
        import torch
        import whisperx
        from pyannote.audio import Pipeline

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "float32"
        hf_token = os.environ.get("HF_TOKEN", "")

        print(f"Device: {self.device}")
        print("Loading Whisper tiny...")
        self.tiny_model = whisperx.load_model("tiny", self.device, compute_type="float32")

        print("Loading WhisperX large-v3...")
        self.whisperx_model = whisperx.load_model("large-v3", self.device, compute_type=self.compute_type)

        print("Loading GigaAM v3...")
        import gigaam
        self.gigaam_model = gigaam.load_model("v3_e2e_rnnt")

        print("Loading pyannote diarization...")
        self.diarize_model = None
        if hf_token:
            try:
                os.environ["HF_HUB_OFFLINE"] = "0"
                self.diarize_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token,  # pyannote 3.1.1 uses use_auth_token
                ).to(torch.device(self.device))
                os.environ["HF_HUB_OFFLINE"] = "1"
                print("Diarization model loaded.")
            except Exception as e:
                print(f"Warning: diarization failed to load: {e}")

        print("All models loaded. Worker ready.")

    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, request: dict) -> dict:
        audio_url = request.get("audio_url")
        language = request.get("language", "auto")
        enable_diarization = request.get("enable_diarization", True)
        min_speakers = int(request.get("min_speakers", 1))
        max_speakers = int(request.get("max_speakers", 4))

        if not audio_url:
            return {"error": "audio_url is required"}

        audio_path = None
        try:
            print(f"Downloading: {audio_url[:80]}...")
            audio_path = self._download_audio(audio_url)
            duration = self._get_duration(audio_path)
            print(f"Duration: {duration:.1f}s")

            if language == "auto":
                print("Detecting language...")
                language = self._detect_language(audio_path)
                print(f"Detected: {language}")

            if language in ("russian", "ru"):
                print("Using GigaAM (Russian)")
                segments, lang = self._run_gigaam(audio_path, enable_diarization, min_speakers, max_speakers)
            else:
                print(f"Using WhisperX ({language})")
                segments, lang = self._run_whisperx(audio_path, language, enable_diarization, min_speakers, max_speakers)

            full_text = " ".join(s["text"] for s in segments if s.get("text"))
            formatted_text = self._build_formatted_text(segments) if enable_diarization else full_text

            return {
                "text": full_text,
                "formatted_text": formatted_text,
                "segments": segments,
                "language": lang,
                "duration": duration,
                "word_count": len(full_text.split()),
            }

        except Exception as e:
            print(traceback.format_exc())
            return {"error": str(e), "traceback": traceback.format_exc()}

        finally:
            import os
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

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
            with open(tmp.name, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
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
        except Exception:
            return 0.0

    def _detect_language(self, audio_path):
        import whisperx
        audio = whisperx.load_audio(audio_path)
        audio_30s = audio[:30 * 16000]
        result = self.tiny_model.transcribe(audio_30s)
        return result.get("language", "en")

    def _run_whisperx(self, audio_path, language, enable_diarization, min_speakers, max_speakers):
        import whisperx
        lang = None if language == "auto" else language
        result = self.whisperx_model.transcribe(audio_path, language=lang, batch_size=16)
        detected_lang = result.get("language", "en")

        try:
            model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio_path, self.device, return_char_alignments=False)
        except Exception as e:
            print(f"Alignment skipped: {e}")

        segments = result.get("segments", [])

        if enable_diarization and self.diarize_model:
            try:
                kwargs = {}
                if min_speakers > 1:
                    kwargs["min_speakers"] = min_speakers
                if max_speakers > 1:
                    kwargs["max_speakers"] = max_speakers
                # Convert to wav for pyannote (torchaudio has issues with m4a/mp4)
                wav_path = audio_path + ".wav"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
                    capture_output=True,
                )
                diarize_input = wav_path if os.path.exists(wav_path) else audio_path
                diarize_result = self.diarize_model(diarize_input, **kwargs)
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
                segments = self._assign_speakers(segments, diarize_result)
            except Exception as e:
                print(f"Diarization failed: {e}")

        return [
            {"start": seg.get("start", 0), "end": seg.get("end", 0),
             "text": seg.get("text", "").strip(), "speaker": seg.get("speaker") if enable_diarization else None}
            for seg in segments
        ], detected_lang

    def _run_gigaam(self, audio_path, enable_diarization, min_speakers, max_speakers):
        raw = self.gigaam_model.transcribe_longform(audio_path)
        segments = []
        for chunk in raw:
            start, end = chunk["boundaries"]
            text = chunk["transcription"].strip()
            if text:
                segments.append({"start": float(start), "end": float(end), "text": text})

        if enable_diarization and self.diarize_model and segments:
            try:
                kwargs = {}
                if min_speakers > 1:
                    kwargs["min_speakers"] = min_speakers
                if max_speakers > 1:
                    kwargs["max_speakers"] = max_speakers
                # Convert to wav for pyannote (torchaudio has issues with m4a/mp4)
                wav_path = audio_path + ".wav"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
                    capture_output=True,
                )
                diarize_input = wav_path if os.path.exists(wav_path) else audio_path
                diarize_result = self.diarize_model(diarize_input, **kwargs)
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
                segments = self._assign_speakers(segments, diarize_result)
            except Exception as e:
                print(f"Diarization failed: {e}")
        elif enable_diarization:
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"

        return segments, "russian"

    def _assign_speakers(self, segments, diarize_result):
        speaker_turns = []
        try:
            for turn, _, speaker in diarize_result.itertracks(yield_label=True):
                speaker_turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        except Exception as e:
            print(f"Could not parse diarization result: {e}")
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
                letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                if idx < len(letters):
                    return f"Speaker {letters[idx]}"
            except ValueError:
                pass
        return speaker_id
