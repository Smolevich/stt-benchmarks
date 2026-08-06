"""Импорт RU/EN сэмплов из открытых датасетов в формат этого бенча.

Зачем: ручной набор — 20 голосовых одного человека. Это ловит сценарии, которых нет
нигде (IP-адреса, пути, ключи, шёпот, переключение языков), но для статистики его
мало, и один голос не даёт обобщения. Открытые наборы закрывают ровно эту дыру и
не требуют ничего записывать.

    # параллельный RU/EN: одни и те же предложения на обоих языках
    python3 -m bench.import_hf_samples --source fleurs --language ru --count 100
    python3 -m bench.import_hf_samples --source fleurs --language en --count 100

    # разнообразие дикторов, лицензия CC0 — аудио можно коммитить
    python3 -m bench.import_hf_samples --source commonvoice --language ru --count 100

Пишет wav-файлы в samples/audio/<set_id>/ и манифест samples/manifest_<set_id>.json
в том же формате, что manifest.template.json, — дальше обычный прогон бенча.

Лицензии на момент написания: Common Voice — CC0 (клипы можно класть в репозиторий),
FLEURS — производный от FLoRes, условия уточнить перед коммитом аудио; по умолчанию
скрипт кладёт его в gitignore-каталог.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCES = {
    # n-way параллельный: RU и EN — одни и те же предложения, поэтому разрыв между
    # языками не смешивается с разницей в содержании.
    "fleurs": {"repo": "google/fleurs",
               "config": {"ru": "ru_ru", "en": "en_us"},
               "split": "test", "text_field": "transcription"},
    # 3695 дикторов на русском, CC0.
    "commonvoice": {"repo": "mozilla-foundation/common_voice_17_0",
                    "config": {"ru": "ru", "en": "en"},
                    "split": "test", "text_field": "sentence"},
}

_BAD = re.compile(r"[^\w\s.,!?—-]", re.UNICODE)


def usable(text: str, min_words: int, max_words: int) -> bool:
    """Отсеиваем слишком короткие и слишком длинные реплики, а также строки со
    странными символами: эталон должен быть тем, что реально произнесено."""
    if not text or _BAD.search(text):
        return False
    return min_words <= len(text.split()) <= max_words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SOURCES), required=True)
    ap.add_argument("--language", choices=("ru", "en"), required=True)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--min-words", type=int, default=8)
    ap.add_argument("--max-words", type=int, default=40)
    ap.add_argument("--one-clip-per-speaker", action="store_true",
                    help="не брать двух записей одного диктора — ради разнообразия голосов")
    args = ap.parse_args()

    from datasets import Audio, load_dataset

    spec = SOURCES[args.source]
    set_id = f"{args.source}_{args.language}"
    out_dir = ROOT / "samples" / "audio" / set_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # streaming=True: качаем ровно столько, сколько нужно, а не весь сплит.
    # decode=False: берём сырые байты файла и кладём на диск как есть. Бенчу нужен
    # файл, а не массив, а декодирование в datasets 5.x тянет torchcodec (то есть
    # весь torch) — ради ничего.
    ds = load_dataset(spec["repo"], spec["config"][args.language],
                      split=spec["split"], streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    samples, seen_speakers = [], set()
    for row in ds:
        if len(samples) >= args.count:
            break
        text = (row.get(spec["text_field"]) or "").strip()
        if not usable(text, args.min_words, args.max_words):
            continue
        speaker = str(row.get("client_id") or row.get("speaker_id") or "")
        if args.one_clip_per_speaker and speaker and speaker in seen_speakers:
            continue
        seen_speakers.add(speaker)

        audio = row["audio"]
        blob = audio.get("bytes")
        if not blob:
            continue
        ext = Path(audio.get("path") or "x.wav").suffix or ".wav"
        idx = len(samples) + 1
        sid = f"{args.language}-{set_id}-{idx:03d}"
        path = out_dir / f"{sid}{ext}"
        path.write_bytes(blob)
        samples.append({"id": sid, "language": args.language,
                        "path": str(path.relative_to(ROOT)), "text": text})

    manifest = ROOT / "samples" / f"manifest_{set_id}.json"
    manifest.write_text(json.dumps(
        {"notes": f"Импортировано из {spec['repo']} ({spec['config'][args.language]}, "
                  f"сплит {spec['split']}) скриптом bench/import_hf_samples.py. "
                  f"Эталон — поле {spec['text_field']} датасета, не вычитывался вручную.",
         "samples": samples}, ensure_ascii=False, indent=2))
    print(f"сэмплов: {len(samples)}, дикторов: {len(seen_speakers) - (1 if '' in seen_speakers else 0)}")
    print(f"аудио:    {out_dir.relative_to(ROOT)}/")
    print(f"манифест: {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
