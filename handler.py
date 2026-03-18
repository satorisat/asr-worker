"""
RunPod Serverless Worker — Audio Transcription
Dual-model: GigaAM v3 (Russian) + WhisperX large-v3 (all other languages)

Requirements:
  - HF_TOKEN env var — needed for pyannote diarization and GigaAM longform VAD
    Get token at huggingface.co, accept terms for:
    pyannote/speaker-diarization-3.1
    pyannote/segmentation-3.0

Input:
  {
    "audio_url": "https://...",       # presigned URL to audio file
    "language": "auto",               # "auto", "russian", "english", etc.
    "enable_diarization": true,
    "min_speakers": 1,
    "max_speakers": 4
  }

Output (LemonFox-compatible format):
  {
    "text": "full transcript",
    "formatted_text": "**Speaker A:** ...\n\n**Speaker B:** ...",
    "segments": [{"start": 0.0, "end": 2.5, "text": "...", "speaker": "SPEAKER_00"}],
    "language": "russian",
    "duration": 123.45,
    "word_count": 150
  }
"""

import os

# Force offline mode — use only models baked into the Docker image
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import tempfile
import subprocess
import traceback

import requests
import torch
import runpod
import whisperx

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "float32"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Load all models once at worker startup (cached in memory between requests)
# ---------------------------------------------------------------------------

print(f"Device: {device}, compute_type: {compute_type}")

print("Loading Whisper tiny (language detection)...")
tiny_model = whisperx.load_model("tiny", device, compute_type="float32")

print("Loading WhisperX large-v3...")
whisperx_model = whisperx.load_model("large-v3", device, compute_type=compute_type)

print("Loading GigaAM v3...")
import gigaam
gigaam_model = gigaam.load_model("v3_e2e_rnnt")

print("Loading pyannote diarization pipeline...")
diarize_model = None
if HF_TOKEN:
    try:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
        print("Diarization model loaded.")
    except Exception as e:
        print(f"Warning: diarization model failed to load: {e}")
else:
    print("Warning: HF_TOKEN not set, diarization disabled.")

print("All models loaded. Worker ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_audio(url: str) -> str:
    """Download audio file to a temp path, return the path."""
    # Determine extension from URL (ignore query params)
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


def get_duration(audio_path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
            capture_output=True, text=True,
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return 0.0


def detect_language(audio_path: str) -> str:
    """Detect language from first 30s using Whisper tiny."""
    audio = whisperx.load_audio(audio_path)
    audio_30s = audio[:30 * 16000]  # 16kHz
    result = tiny_model.transcribe(audio_30s)
    return result.get("language", "en")


def format_speaker_name(speaker_id: str) -> str:
    if speaker_id.startswith("SPEAKER_"):
        try:
            idx = int(speaker_id.replace("SPEAKER_", ""))
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if idx < len(letters):
                return f"Speaker {letters[idx]}"
        except ValueError:
            pass
    return speaker_id


def build_formatted_text(segments: list) -> str:
    """Build **Speaker A:** ... formatted text from segments."""
    lines = []
    current_speaker = None
    current_texts = []

    for seg in segments:
        speaker = seg.get("speaker") or "SPEAKER_00"
        text = seg.get("text", "").strip()
        if not text:
            continue
        if speaker != current_speaker:
            if current_speaker and current_texts:
                lines.append(f"**{format_speaker_name(current_speaker)}:** {' '.join(current_texts)}")
            current_speaker = speaker
            current_texts = [text]
        else:
            current_texts.append(text)

    if current_speaker and current_texts:
        lines.append(f"**{format_speaker_name(current_speaker)}:** {' '.join(current_texts)}")

    return "\n\n".join(lines)


def assign_speakers_to_segments(segments: list, diarize_result) -> list:
    """Assign pyannote speaker labels to segments by max overlap."""
    speaker_turns = []
    try:
        for turn, _, speaker in diarize_result.itertracks(yield_label=True):
            speaker_turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    except Exception as e:
        print(f"Could not parse diarization result: {e}")
        return segments

    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for turn in speaker_turns:
            overlap = min(seg_end, turn["end"]) - max(seg_start, turn["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        seg["speaker"] = best_speaker

    return segments


# ---------------------------------------------------------------------------
# Transcription functions
# ---------------------------------------------------------------------------

def run_whisperx(audio_path: str, language: str, enable_diarization: bool,
                 min_speakers: int, max_speakers: int) -> tuple[list, str]:
    lang = None if language == "auto" else language

    result = whisperx_model.transcribe(audio_path, language=lang, batch_size=16)
    detected_lang = result.get("language", "en")

    # Word-level alignment (improves timestamp accuracy)
    try:
        model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio_path, device,
            return_char_alignments=False,
        )
    except Exception as e:
        print(f"Alignment skipped: {e}")

    segments = result.get("segments", [])

    # Speaker diarization
    if enable_diarization and diarize_model:
        try:
            kwargs = {}
            if min_speakers > 1:
                kwargs["min_speakers"] = min_speakers
            if max_speakers > 1:
                kwargs["max_speakers"] = max_speakers
            diarize_segments = diarize_model(audio_path, **kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            segments = result.get("segments", [])
        except Exception as e:
            print(f"Diarization failed: {e}")

    normalized = [
        {
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": seg.get("text", "").strip(),
            "speaker": seg.get("speaker") if enable_diarization else None,
        }
        for seg in segments
    ]

    return normalized, detected_lang


def run_gigaam(audio_path: str, enable_diarization: bool,
               min_speakers: int, max_speakers: int) -> tuple[list, str]:
    # transcribe_longform splits audio via VAD and transcribes each chunk
    raw = gigaam_model.transcribe_longform(audio_path)

    segments = []
    for chunk in raw:
        start, end = chunk["boundaries"]
        text = chunk["transcription"].strip()
        if text:
            segments.append({"start": float(start), "end": float(end), "text": text})

    # Speaker diarization
    if enable_diarization and diarize_model and segments:
        try:
            kwargs = {}
            if min_speakers > 1:
                kwargs["min_speakers"] = min_speakers
            if max_speakers > 1:
                kwargs["max_speakers"] = max_speakers
            diarize_result = diarize_model(audio_path, **kwargs)
            segments = assign_speakers_to_segments(segments, diarize_result)
        except Exception as e:
            print(f"Diarization failed: {e}")
    elif enable_diarization:
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"

    return segments, "russian"


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

def handler(job):
    job_input = job.get("input", {})

    audio_url = job_input.get("audio_url")
    language = job_input.get("language", "auto")
    enable_diarization = job_input.get("enable_diarization", True)
    min_speakers = int(job_input.get("min_speakers", 1))
    max_speakers = int(job_input.get("max_speakers", 4))

    if not audio_url:
        return {"error": "audio_url is required"}

    audio_path = None
    try:
        print(f"Downloading: {audio_url[:80]}...")
        audio_path = download_audio(audio_url)
        duration = get_duration(audio_path)
        print(f"Duration: {duration:.1f}s")

        # Language detection
        if language == "auto":
            print("Detecting language...")
            language = detect_language(audio_path)
            print(f"Detected: {language}")

        # Route to correct model
        if language in ("russian", "ru"):
            print("Using GigaAM (Russian)")
            segments, lang = run_gigaam(audio_path, enable_diarization, min_speakers, max_speakers)
        else:
            print(f"Using WhisperX ({language})")
            segments, lang = run_whisperx(audio_path, language, enable_diarization, min_speakers, max_speakers)

        full_text = " ".join(s["text"] for s in segments if s.get("text"))
        formatted_text = build_formatted_text(segments) if enable_diarization else full_text

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
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


runpod.serverless.start({"handler": handler})
