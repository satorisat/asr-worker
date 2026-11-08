"""
РАЗОВАЯ укладка весов многоязычной модели в том Modal. НЕ часть рабочего пути воркера.

Зачем отдельный файл: библиотека GigaAM качает веса с CDN Сбера, а у него с 04.11.2026
просрочен сертификат (проверено `openssl s_client`: Let's Encrypt YR1,
06.08.2026–04.11.2026 — не санкции, а несработавшее автопродление). Поэтому веса
кладём в том заранее, а воркер их только читает: `_download_file` библиотеки возвращает
готовый путь, если файл на месте, и в сеть не идёт.

Почему проверка TLS здесь снята, а в воркере её снятия НЕТ:
  * подлинность файла закрывает MD5 из самой библиотеки (`gigaam._MODEL_HASHES`),
    полученный по живому TLS с GitHub — для подмены нужен ПРООБРАЗ MD5, не коллизия;
  * тот же MD5 библиотека сверяет при КАЖДОЙ загрузке модели
    (`assert hash_path(path) == _MODEL_HASHES[name]` в `load_model`), то есть битый
    файл роняет контейнер, а не уезжает в работу молча;
  * этот файл запускается руками (`modal run`), в рабочем пути воркера не участвует.
Когда Сбер продлит сертификат — снять `--insecure` и запускать с обычной проверкой.

Зеркало HuggingFace (`ai-sage/GigaAM-Multilingual@large_ctc`) НЕ подходит: проверено
скачиванием — там плоский словарь тензоров (`model.encoder.layers.0...`) без ключа
`cfg`, из которого библиотека строит модель, и MD5 другой
(c77570645840c876f173ac8e14e71ef9 вместо 79a9adde50dd7f35bbf70927cb6557d0). Веса те же
2233 МБ, упаковка другая — чтобы её использовать, пришлось бы писать свой путь
инференса вместо `transcribe_longform`, который уже работает.

Запуск (сначала укладка, ПОТОМ деплой воркера):
  modal run provision_ml_weights.py                 # обычная проверка сертификата
  modal run provision_ml_weights.py --insecure      # пока сертификат CDN просрочен
  modal run provision_ml_weights.py --check         # только проверить, что уже лежит
"""

import json
import os

import modal

app = modal.App("asr-ml-provision")

volume = modal.Volume.from_name("asr-models-cache-ml", create_if_missing=True)
CACHE_DIR = "/vol/hf_cache"
MODEL_NAME = "multilingual_large_ctc"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "requests",
        "huggingface_hub>=0.20.0,<1.0",
        "soundfile",
        "scipy",
        "sentencepiece",
        "git+https://github.com/salute-developers/GigaAM.git",
    )
)


def _md5_file(path: str) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


@app.function(image=image, cpu=2.0, memory=8192, volumes={CACHE_DIR: volume}, timeout=3600)
def provision(insecure: bool = False, check_only: bool = False) -> dict:
    import ssl
    import time
    import urllib.request

    import gigaam

    want = gigaam._MODEL_HASHES[MODEL_NAME]
    root = os.path.join(CACHE_DIR, "gigaam")
    os.makedirs(root, exist_ok=True)
    final = os.path.join(root, f"{MODEL_NAME}.ckpt")
    out = {"model": MODEL_NAME, "path": final, "want_md5": want}

    if os.path.exists(final):
        out["present"] = True
        out["md5"] = _md5_file(final)
        out["md5_ok"] = out["md5"] == want
        out["bytes"] = os.path.getsize(final)
        if not out["md5_ok"]:
            # Выходить с нулём тут нельзя: это шаг перед деплоем, и «веса на месте»
            # при битом файле означало бы зелёный свет на деплой воркера, который
            # потом падает на проверке MD5 внутри библиотеки при каждом старте.
            print("ВЕСА НА МЕСТЕ, НО БИТЫЕ:", json.dumps(out, ensure_ascii=False))
            raise RuntimeError(
                f"MD5 не совпал у {final}: {out['md5']} != {want}. "
                f"Удалить файл осознанно и уложить заново."
            )
        print("веса на месте и целые:", json.dumps(out, ensure_ascii=False))
        return out

    out["present"] = False
    stale = final + ".part"
    if os.path.exists(stale):
        # Жёсткое убийство контейнера (OOM, вытеснение, таймаут) обработчик не
        # выполняет, поэтому огрызок переживает прогон. Валидными весами его никто не
        # примет (имя другое), но место он занимает — сообщаем.
        out["stale_part_bytes"] = os.path.getsize(stale)
    if check_only:
        print("весов нет:", json.dumps(out, ensure_ascii=False))
        raise RuntimeError(f"весов нет: {final} — запустить без --check, чтобы уложить")

    url = f"https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/{MODEL_NAME}.ckpt"
    ctx = None
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        print("ВНИМАНИЕ: проверка сертификата CDN снята, подлинность проверяется по MD5")

    tmp = final + ".part"
    t0 = time.perf_counter()
    size = 0
    req = urllib.request.Request(url, headers={"User-Agent": "audiovtext-provision"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=1800) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 22)
                if not chunk:
                    break
                fh.write(chunk)
                size += len(chunk)
    except Exception:
        # Недокачанный файл не оставляем: он не совпадёт по MD5 и будет мешать разбору.
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError as exc:
                print(f"не удалось убрать {tmp}: {exc}")
        raise

    out["url"] = url
    out["bytes"] = size
    out["seconds"] = round(time.perf_counter() - t0, 1)
    out["md5"] = _md5_file(tmp)
    out["md5_ok"] = out["md5"] == want

    if not out["md5_ok"]:
        # Сначала печатаем диагноз, потом удаляем: ошибка самого удаления не должна
        # подменить сообщение о несовпадении MD5.
        print("MD5 НЕ СОВПАЛ, файл удаляется:", json.dumps(out, ensure_ascii=False))
        try:
            os.unlink(tmp)
        except OSError as exc:
            print(f"не удалось удалить {tmp}: {exc}")
        raise RuntimeError(f"MD5 mismatch: {out['md5']} != {want}")

    os.rename(tmp, final)
    out["installed"] = True
    # Фоновые коммиты Modal персистят запись и сами, но это разовая задача: контейнер
    # закончится сразу после возврата, поэтому фиксируем явно и ждём результат.
    volume.commit()
    print("уложено:", json.dumps(out, ensure_ascii=False))
    return out


@app.local_entrypoint()
def main(insecure: bool = False, check: bool = False):
    print(json.dumps(provision.remote(insecure, check), ensure_ascii=False, indent=2))
