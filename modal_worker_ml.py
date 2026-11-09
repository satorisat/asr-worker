"""
Modal Serverless Worker — многоязычная транскрибация: казахский, киргизский, узбекский
GigaAM multilingual_large_ctc + pyannote speaker diarization

Копия modal_worker.py: та же структура, тот же образ, та же фаза снимка, тот же
HTTP-слой. Меняется модель — `multilingual_large_ctc` вместо `v3_e2e_rnnt`. У неё
CTC-голова и посимвольный словарь, языкового параметра нет: один и тот же код пишет
казахский и киргизский кириллицей, узбекский латиницей. Какой язык сюда отправлять,
решает бэкенд; поле `language` из запроса возвращается в ответе как есть. Пунктуации и
заглавных эта модель не ставит. Токенизатор для неё не качается — библиотека берёт его
только для `v1_rnnt` и моделей с `e2e` в имени (проверено по её коду).

Парный воркер на Beam — `gigaam_ml_worker.py` в репозитории whisper-beam-fallback.

Веса качает сам воркер в общий том, как сосед. Один нюанс: библиотека берёт их с CDN
Сбера, а у него с 04.11.2026 просрочен сертификат (проверено openssl s_client: Let's
Encrypt YR1, 06.08.2026–04.11.2026 — не санкции, а несработавшее автопродление) и
качает через `urllib.request.urlopen` без своего ssl-контекста. Поэтому на время
загрузки весов подменяется контекст по умолчанию и возвращается сразу после; pyannote и
HuggingFace грузятся с полной проверкой. Целостность держит сама библиотека:
`load_model` делает `assert hash_path(path) == _MODEL_HASHES[name]` при КАЖДОЙ
загрузке, то есть битый или подменённый чекпоинт роняет контейнер, а не уезжает в
работу; для подмены нужен прообраз MD5, а не коллизия. Зеркало
`ai-sage/GigaAM-Multilingual@large_ctc` на HuggingFace не подходит: там плоский словарь
тензоров без ключа `cfg` (проверено скачиванием). Когда сертификат продлят, подмену
можно убрать без других правок.

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

# Persistent volume for HF model cache — avoids re-downloading on every cold start
volume = modal.Volume.from_name("asr-models-cache", create_if_missing=True)
CACHE_DIR = "/vol/hf_cache"
MODEL_NAME = "multilingual_large_ctc"   # kk / ky / uz, CTC-голова

# Логи и кусочный кеш клиента xet — ВНЕ тома.
#
# Дока Modal про снапшоты памяти говорит прямо: «Deleting files in a Volume used
# during restore will cause restore failures». Именно это и происходило. По доке HF
# `HF_XET_CACHE` по умолчанию равен `$HF_HOME/xet`, то есть ложился в том, и клиент
# писал туда файлы с меткой времени (`xet/logs/xet_<метка>_N.log`), а старые сам
# подчищал (в томе 358 логов, самый старый — за две недели). GigaAM снимает снапшот
# ПОСЛЕ загрузки моделей (`enter(snap=True)`), поэтому ссылка на такой лог попадала в
# снапшот; когда лог подчищался, восстановление падало намертво и уже не могло
# подняться до следующего деплоя:
#   vfs.CompleteRestore() failed ... failed to walk ".../xet/logs/xet_...log"
#   Runner failed with exit code: 128
# 12.10 так упало четыре контейнера — все по ссылке на лог от 25.09. Каждое падение
# стоит лишнего холодного старта. Том общий на три воркера (его может подчистить любой
# из них), поэтому переменная ставится во всех трёх.
#
# На офлайн это НЕ влияет: веса в томе лежат отдельно — `gigaam/` (свой `download_root`
# у библиотеки GigaAM, см. `gigaam.load_model(..., download_root=…)`) и
# `whisper/` (download_root у faster-whisper); каталог `xet/` при `HF_HUB_OFFLINE=1`
# для загрузки весов не нужен вовсе, там оставались только логи и кусочный кеш
# докачки. Порядок загрузки моделей и сам `HF_HOME` не тронуты.
#
# Задаётся на уровне модуля — до импорта huggingface_hub, чтобы значение не зависело
# от того, читает библиотека переменную при импорте или в момент обращения. Образ не
# меняется, значит пересборки нет.
os.environ.setdefault("HF_XET_CACHE", "/tmp/hf_xet")

MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


PUNCT_REPO = "Horizon-Labs/punctuation-restoration-small"
# Окно 400 токенов с перехлёстом 50 — как в эталонном punctuate.py автора модели.
# Проверено прогоном на 3000 слов: при окне 2048 знаков меньше (263 против 318) и точки
# на концах предложений теряются, расхождение между окнами 44%. Выигрыш по времени
# (0,19с против 0,93с на 3000 слов) того не стоит.
PUNCT_CHUNK, PUNCT_OVERLAP = 400, 50


def _load_punct(device):
    """Модель расстановки знаков. Грузится ВМЕСТЕ с остальными, в фазе снимка
    (@modal.enter(snap=True)), поэтому её 7-8с попадают в снимок памяти и на запросы не
    влияют вовсе — к концу расшифровки модель уже готова. Сам проход по тексту — 1,4с
    на час речи на A10.

    Возвращает (модель, токенизатор). Исключения НЕ глотает — их гасит вызывающий,
    чтобы обёртка offline-first могла сделать повторную попытку."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PUNCT_REPO)
    # reference_compile=False обязателен: иначе transformers зовёт torch.compile, а в
    # образе нет компилятора C — падает с «Failed to find C compiler» (проверено).
    model = AutoModelForTokenClassification.from_pretrained(
        PUNCT_REPO,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        reference_compile=False,
    ).eval().to(device)
    print(f"Punctuation model loaded ({PUNCT_REPO}, {device})")
    return model, tok


def _punctuate(punct, text: str) -> str:
    """Расставить знаки в тексте одного сегмента. Слова не меняются и не добавляются:
    модель только помечает, после какого слова ставить знак (классификация токенов).
    Регистр не трогаем — у этой модели в наборе метки только знаки."""
    import torch

    model, tok = punct
    # truncation=False обязателен: в tokenizer.json включена обрезка на 512 токенов, с
    # ней длинный текст молча остаётся без знаков после первых ~350 слов (проверено).
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if not ids:
        return text
    # ключи id2label у transformers ЦЕЛЫЕ (в config.json строки) — приводим.
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    device = next(model.parameters()).device
    marks, start = {}, 0
    with torch.no_grad():
        while start < len(ids):
            end = min(len(ids), start + PUNCT_CHUNK)
            x = torch.tensor([[tok.cls_token_id] + ids[start:end] + [tok.sep_token_id]], device=device)
            pred = model(input_ids=x).logits[0, 1:-1].argmax(-1).tolist()
            keep_to = end if end == len(ids) else end - PUNCT_OVERLAP
            for k in range(start, keep_to):
                s_off, e_off = offs[k]
                if e_off > s_off:
                    lab = id2label[pred[k - start]]
                    if lab != "0":
                        marks[e_off] = lab
            if end == len(ids):
                break
            start = keep_to
    # знак ставим только на границе слова, чтобы не разорвать слово
    res, prev = [], 0
    for e in sorted(marks):
        if not (e == len(text) or text[e] == " "):
            continue
        res.append(text[prev:e] + marks[e])
        prev = e
    return "".join(res) + text[prev:]


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


def _hf_offline(on: bool):
    v = "1" if on else "0"
    os.environ["HF_HUB_OFFLINE"] = v
    os.environ["TRANSFORMERS_OFFLINE"] = v


def _load_offline_first(loader, what: str):
    """Load from the volume HF cache with NO network; on a genuine cache miss (first-ever
    container / wiped cache) fall back to a one-time online download, then re-assert offline.
    Keeps a HuggingFace outage from breaking cold starts once the model is cached."""
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


# ---------------------------------------------------------------------------
# Image — GigaAM + pyannote
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "git")
    # torch первым — чтобы torchcodec (dep pyannote 3.3) нашёл правильные CUDA-колёса.
    # Пин 2.8.0 — как в whisper-воркере (тот же Modal-стек, работает): без пина pip брал
    # свежий torchaudio 2.9, где удалён AudioMetaData → падал `from pyannote.audio import Pipeline`.
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
        "pyannote.audio==3.3.2",  # 3.3.2 = последний 3.x: numpy 2.x safe, без torchcodec
        "speechbrain>=1.0.0",
        "scikit-learn>=1.3.0",
        # transformers нужен ТОЛЬКО для модели пунктуации (ModernBERT): архитектура
        # появилась в 4.48, на более старых загрузка падает с «does not recognize this
        # architecture» (проверено прогоном на 4.46.3). Pyannote его не тянет — он нужен
        # ему лишь для необязательного набора separation, который мы не ставим.
        "transformers==4.49.0",
        "git+https://github.com/salute-developers/GigaAM.git",
    )
    # Кеш pyannote и huggingface_hub — в том, а не в контейнер.
    # Почему на уровне образа, а не в рантайме: pyannote читает
    # CACHE_DIR = os.getenv("PYANNOTE_CACHE", "~/.cache/torch/pyannote") на уровне модуля,
    # а huggingface_hub вычисляет HF_HOME/HF_HUB_CACHE при импорте. Присваивание
    # os.environ[...] в load_models происходит ПОСЛЕ импортов, то есть для них уже поздно —
    # поэтому веса pyannote до сих пор качались заново на каждом холодном старте
    # (авария 16.07.2026, когда HuggingFace был недоступен).
    # ВНИМАНИЕ: это было верно, пока менялся только слой .env(). Сейчас в pip_install
    # добавлен transformers, значит ЭТОТ слой пересоберётся, и вместе с ним заново
    # разрешатся все незапиненные строки списка — включая git+GigaAM.git на HEAD.
    # На HEAD у GigaAM сейчас onnx==1.19.*, onnxruntime==1.23.*, numpy==2.*, а набор
    # longform целит в pyannote 4 / torch 2.10 / transformers 5 (проверено по pyproject).
    # Мы ставим пакет без наборов, но пины ядра приедут. Перед деплоем это стоит
    # учитывать: на Beam такая же пересборка прошла зелёной, на Modal она ещё не делалась.
    # HF_XET_CACHE здесь ОБЯЗАТЕЛЕН: как только HF_HOME указывает в том, путь кеша xet по
    # умолчанию становится $HF_HOME/xet — ровно тот каталог, из-за которого 12.10 ломалось
    # восстановление снимка. Без этой строки защита держится на одном setdefault и порядке
    # импортов.
    # volume.commit() сознательно НЕ добавлен: Modal фиксирует том в фоне и при остановке
    # контейнера, а явный коммит в фазе snap=True — это та самая запись в том, на которой
    # ломалось восстановление.
    .env({
        "HF_HOME": CACHE_DIR,
        "PYANNOTE_CACHE": CACHE_DIR + "/pyannote",
        "HF_XET_CACHE": "/tmp/hf_xet",
    })
)


# ---------------------------------------------------------------------------
# Worker class
# ---------------------------------------------------------------------------

@app.cls(
    # Список карт, а не одна: Modal берёт первую СВОБОДНУЮ, порядок соблюдает
    # («Modal allows specifying a list of possible GPU types… respects the ordering»).
    # Зачем: 21.10.2026 за 12 часов 83 задания, работа p50 15с, а полное время p50 233с
    # и p90 1057с — при бюджете опроса 900с это значит, что каждое десятое задание
    # обязано провалиться на резерв. Залипаний сети при этом НОЛЬ (все события
    # `wedged_attempts=0`), контейнер поднимается за 4,4с, лимитов у нас не задано,
    # план Starter даёт «100 containers + 10 GPU concurrency» — то есть упираемся не в
    # себя и не в сеть, а в отсутствие свободной карты ЗАПРОШЕННОГО типа.
    #
    # Цена: за задание в 15с L4 $0,0033, A10 $0,0046, L40S $0,0081, A100 $0,0087,
    # H100 $0,0165 — но дорогие берутся только когда дешёвых нет, а H100 ту же работу
    # делает быстрее, поэтому разница на деле меньше часовой ставки. Дороже на цент
    # против 15 минут ожидания и второго прогона на Beam — обмен выгодный.
    #
    # T4 намеренно НЕ включён (решение юзера 21.10): он втрое медленнее на fp32, и
    # 45с × 0,000164 ≈ $0,0074 за задание, то есть как L40S, только медленно.
    # По памяти подходят все: модель ~1 млрд параметров + pyannote, это единицы ГБ.
    # По точности тоже: bf16 нигде нет (у Whisper float16, у GigaAM fp32).
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

    # snap=True: runs BEFORE the memory snapshot. GPU is NOT available in this phase, so we
    # load GigaAM and pyannote on CPU — that loaded state is captured in the snapshot. The
    # CPU→GPU move happens in move_to_gpu (snap=False) after restore.
    @modal.enter(snap=True)
    def load_models(self):
        import time
        import torch
        from pyannote.audio import Pipeline
        import gigaam

        self._load_error = None
        self.device = "cpu"  # real device is set in move_to_gpu (GPU unavailable during snapshot)

        os.environ["HF_HOME"] = CACHE_DIR
        _hf_offline(True)
        # GigaAM хранит веса в torch.hub dir — направляем в volume
        torch.hub.set_dir(CACHE_DIR)

        hf_token = os.environ.get("HF_TOKEN", "")

        # Patch GigaAM vad_utils: passes local snapshot path to Model.from_pretrained()
        # but pyannote 3.3.2 expects a HF repo ID. Override to pass repo ID directly.
        import gigaam.vad_utils as _vad
        def _patched_load_segmentation_model(model_id):
            from pyannote.audio import Model
            from torch.torch_version import TorchVersion
            from pyannote.audio.core.task import Problem, Resolution, Specifications
            def _load():
                with torch.serialization.safe_globals([TorchVersion, Problem, Specifications, Resolution]):
                    return Model.from_pretrained(model_id, use_auth_token=os.environ.get("HF_TOKEN"))
            # Called lazily by transcribe_longform's VAD (post-restore, offline). Route through
            # offline-first so it self-heals on a cold cache instead of hard-failing offline.
            return _load_offline_first(_load, f"VAD segmentation ({model_id})")
        _vad.load_segmentation_model = _patched_load_segmentation_model

        # Download GigaAM weights into the persistent Modal Volume (Modal-recommended
        # pattern for model weights, mirrors what Whisper already does). First successful
        # container skills the volume; all subsequent cold starts read from volume with no
        # network. On corrupt/partial file we wipe the cached path and retry with backoff.
        gigaam_cache = os.path.join(CACHE_DIR, "gigaam")
        os.makedirs(gigaam_cache, exist_ok=True)

        print(f"Loading GigaAM {MODEL_NAME} (CPU, snapshot phase)...")
        # Проверка сертификата CDN снимается ТОЛЬКО на загрузку весов и
        # возвращается в finally — причина в шапке файла. Цикл повторов, его else и
        # volume.commit() ниже оставлены как в modal_worker.py.
        import ssl
        _prev_ssl_context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            last_err = None
            downloaded_fresh = False
            for attempt in range(1, 6):
                try:
                    # Пробуем СВОЙ файл, а не каталог: он общий с русским воркером, там
                    # всегда лежат его веса, и проба по каталогу делала бы
                    # volume.commit() ниже мёртвым кодом.
                    had_files_before = os.path.exists(os.path.join(gigaam_cache, f"{MODEL_NAME}.ckpt"))
                    # CPU load only — no .to(cuda) here (GPU unavailable in the snapshot phase).
                    self.gigaam_model = gigaam.load_model(MODEL_NAME, download_root=gigaam_cache)
                    if not had_files_before:
                        downloaded_fresh = True
                    break
                except Exception as e:
                    last_err = e
                    print(f"GigaAM load attempt {attempt}/5 failed: {type(e).__name__}: {e}")
                    # Чистим ТОЛЬКО свой чекпоинт, а не каталог целиком, как в
                    # modal_worker.py: том общий, download_root у обоих воркеров один и тот
                    # же, а у русского воркера обхода просроченного сертификата CDN нет —
                    # rmtree каталога снёс бы его веса без возможности их скачать. Смысл
                    # чистки сохранён: недокачанный или битый файл этой модели убирается.
                    ckpt = os.path.join(gigaam_cache, f"{MODEL_NAME}.ckpt")
                    if os.path.exists(ckpt):
                        try:
                            os.unlink(ckpt)
                            print(f"removed {ckpt} before retry")
                        except OSError as exc:
                            print(f"could not remove {ckpt}: {exc}")
                    os.makedirs(gigaam_cache, exist_ok=True)
                    if attempt < 5:
                        time.sleep(min(2 ** attempt, 30))
            else:
                # Don't swallow into _load_error: that keeps the broken container alive
                # returning errors to every request. Raise so Modal kills it and spawns fresh.
                print(f"FATAL: GigaAM failed to load after 5 attempts:\n{traceback.format_exc()}")
                raise RuntimeError(f"GigaAM load failed: {type(last_err).__name__}: {last_err}")
        finally:
            ssl._create_default_https_context = _prev_ssl_context
        print("GigaAM loaded (CPU)")

        # If this container was the one that populated the volume, commit so that
        # future cold starts see the cached weights without re-downloading.
        if downloaded_fresh:
            try:
                volume.commit()
                print("GigaAM weights committed to volume")
            except Exception as e:
                print(f"Warning: volume.commit() failed: {e}")

        # pyannote on CPU, offline-first from the volume cache (was HF_HUB_OFFLINE=0 → online
        # every cold start, which broke on the 2026-07-16 HF outage). Online only as a
        # one-time fallback on a genuine cache miss. GPU move happens in move_to_gpu.
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

        # Пунктуация: эта линия GigaAM знаков не ставит (посимвольная CTC-голова),
        # поэтому ставим их отдельной моделью. Грузим ЗДЕСЬ же, в фазе снимка, на
        # процессоре — карты в этой фазе нет; перенос на карту идёт в move_to_gpu
        # вместе с остальными моделями. Так загрузка (7-8с) попадает в снимок памяти и
        # на запросы не влияет: к концу расшифровки модель уже готова.
        print("Loading punctuation model (CPU, snapshot phase)...")
        # Через _load_offline_first, как GigaAM и pyannote: сам _hf_offline сейчас не
        # действует (переменные ставятся после импорта huggingface_hub), но если его
        # когда-нибудь починят, прямая загрузка молча отключила бы пунктуацию на
        # холодном кеше — а здесь будет честная попытка в сети.
        try:
            self.punct = _load_offline_first(lambda: _load_punct("cpu"), "punctuation model")
        except Exception as e:
            # Гасим ЗДЕСЬ, а не внутри _load_punct: иначе обёртка offline-first
            # бессмысленна — повторная попытка не сработает, если загрузчик сам съедает
            # исключение.
            print(f"Warning: punctuation model failed to load: {type(e).__name__}: {e}")
            self.punct = None

        print("Snapshot-phase load complete (models on CPU).")

    # snap=False: runs AFTER snapshot restore (GPU available). Move the CPU-loaded models to
    # GPU. Fast — weights are already in RAM from the snapshot, only the CPU→GPU transfer runs.
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
                # Don't leave it on CPU silently — diarization would then run at CPU speed on
                # a GPU request path and risk a timeout. Disable it so behavior matches
                # "diarization unavailable" (diarization_available:false) honestly.
                print(f"Warning: moving diarization to GPU failed ({e}) — disabling diarization")
                self.diarize_model = None
        if getattr(self, "punct", None) is not None:
            try:
                model, tok = self.punct
                self.punct = (model.to(self.device).half(), tok)
            except Exception as e:
                # Отключаем, а НЕ «оставляем на процессоре»: .to() и .half() меняют модуль
                # на месте, поэтому после сбоя на середине он остаётся полуперенесённым,
                # и _punctuate берёт устройство у первого параметра — каждый сегмент падал
                # бы на несовпадении устройств. Так же устроена ветка диаризации выше.
                print(f"Warning: moving punctuation to GPU failed ({e}) — punctuation disabled")
                self.punct = None
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

            # Знаки — ПО СЕГМЕНТАМ, а не по склеенному тексту: сегмент это реплика от
            # VAD со своими временами и меткой говорящего, так они гарантированно не
            # поедут. Отказ пунктуации не роняет расшифровку — текст отдаётся как есть.
            punct = getattr(self, "punct", None)
            if punct and segments:
                import time as _t
                t0 = _t.perf_counter()
                n_changed = 0
                for seg in segments:
                    raw = seg.get("text") or ""
                    if not raw:
                        continue
                    try:
                        marked = _punctuate(punct, raw)
                    except Exception as e:
                        print(f"Warning: punctuation failed ({type(e).__name__}: {e}) — "
                              f"returning the rest of the text without marks")
                        break
                    if marked != raw:
                        seg["text"] = marked
                        n_changed += 1
                print(f"Punctuation: {n_changed}/{len(segments)} segments in "
                      f"{_t.perf_counter() - t0:.2f}s")
            elif not punct:
                print("Punctuation model unavailable — returning text without marks")

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
        result = self.gigaam_model.transcribe_longform(input_path)
        # GigaAM changed transcribe_longform's return type across versions: newer releases
        # return a LongformTranscriptionResult of Segment dataclasses (.text/.start/.end);
        # older ones returned dicts ({"transcription","boundaries"}). Support both so a
        # GigaAM version bump doesn't break us (mirrors the Beam gigaam worker).
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

# Отдельный ЛЁГКИЙ образ для HTTP-слоя ниже. Он только принимает задание, вызывает
# `spawn` и отдаёт результат — внутри `web()` используются лишь `os`, `modal` и
# `fastapi`, ничего из тяжёлого стека. Прежде слой приёма поднимался из того же
# образа, что GPU-класс: база `nvidia/cuda:12.1.1-cudnn8-runtime` плюс torch,
# torchaudio, pyannote, speechbrain. Отсюда баннер CUDA в логе ВЕБ-функции
# (видно 12.10.2026) и лишние секунды на каждый подъём слоя приёма.
#
# У нас `@modal.asgi_app()`, и для него дока Modal состава образа не диктует — ASGI-
# приложение может быть любым, fastapi нужен просто потому, что его импортирует наш
# код. Требование «the Image your Function uses must have FastAPI installed» стоит в
# доке у соседнего `@modal.fastapi_endpoint`; пример там приведён ровно такой же:
# `modal.Image.debian_slim().pip_install("fastapi[standard]")`.
#
# GPU-образ НЕ пересобирается: по доке «If the definition hasn't changed since the
# last time you ran or deployed your App» он берётся из кеша, а его определение мы
# не трогаем. Собирается только этот новый, маленький.
#
# Python 3.10 — та же минорная серия, что у GPU-образа (`add_python="3.10"`). Именно
# серия, а не тот же сборочный вариант: `add_python` ставит python-build-standalone, а
# `debian_slim` — официальный `python:3.10-slim`. На передачу данных это не влияет:
# через `spawn` уходит плоский dict и такой же возвращается.
# Версия запинена: проверена прогоном (пустое тело -> наш 400, с audio_url -> 200,
# fastapi 0.143.0 / starlette 1.7.0 / pydantic 2.14.1). Прежде HTTP-слой ехал на той
# версии, что запеклась в давно закешированный CUDA-образ; у нового образа она
# разрешалась бы заново на каждой сборке, и смена правил разбора тела запроса
# (у нас это голый `request: dict`) дала бы 422 на каждую отправку при зелёном
# деплое. Пин — как у остального несущего в репозитории (torch, pyannote).
web_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "fastapi[standard]==0.143.0"
)


@app.function(
    image=web_image,
    # 300с, а не 2: это HTTP-слой (приём задания и отдача результата), а не GPU.
    # Бэкенд опрашивает результат раз в 5 секунд, поэтому при двух секундах контейнер
    # успевал умереть МЕЖДУ опросами и поднимался заново почти на каждый запрос —
    # видно в логах Modal 12.10.2026: баннер CUDA в логе именно веб-функции посреди
    # окна опроса. По доке допустимый диапазон 2с…20мин, значение по умолчанию 60с.
    # Цена простоя при запросе по умолчанию (0,125 ядра, 128 МиБ) — порядка $0,26 в
    # месяц на нашем потоке; GPU-классы это не затрагивает, у них окно остаётся 2с,
    # потому что простой L4 стоит $0,80 в час.
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
            # spawn.aio, а не spawn: эндпоинт асинхронный, и Modal сам предупреждает в
            # логе («A blocking Modal interface is being used in an async context»,
            # подсказанная им форма — `await worker.do_transcribe.spawn.aio(request)`).
            # Дока: «The async interface can be accessed by appending `.aio`».
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
