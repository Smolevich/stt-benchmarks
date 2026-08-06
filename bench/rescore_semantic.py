"""Пересчёт WER/CER поверх уже собранных транскриптов, с семантической нормализацией.

Ничего не запускает и никуда не ходит: берёт reference/hypothesis из
results/consolidated.json, поэтому стоит ноль и воспроизводится офлайн.

    python3 bench/rescore_semantic.py --audio-set live
    python3 bench/rescore_semantic.py --audio-set live --language ru --show-deltas
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from bench.normalize_semantic import semantic_normalize
from bench.wer_eval import compute_cer, compute_wer

ROOT = Path(__file__).resolve().parent.parent
CONSOLIDATED = ROOT / "results" / "consolidated.json"
OUT_CSV = ROOT / "results" / "semantic_summary.csv"

# Те же два стресс-сэмпла, что README исключает из «WER clean»: на них обычный WER
# улетает в 60-95% у всех моделей. Смысл упражнения — показать, сколько из этого
# было форматированием, поэтому считаем и с ними, и без них.
STRESS = ("-04-digits", "-07-names")


def is_stress(sample_id: str) -> bool:
    return any(marker in sample_id for marker in STRESS)


def audio_set_of(run: dict) -> str:
    """В consolidated.json набор зашит суффиксом sample_id: «en-01-clean» — живой
    голос, «en-01-clean__elevenlabs_rachel_mixed» — синтетика."""
    sid = run.get("sample_id", "")
    return sid.split("__", 1)[1] if "__" in sid else "live"


def rescore(runs: list[dict]) -> list[dict]:
    out = []
    for r in runs:
        ref, hyp = r.get("reference"), r.get("hypothesis")
        if not ref:
            continue
        ref_n, hyp_n = semantic_normalize(ref), semantic_normalize(hyp)
        out.append({**r,
                    "audio_set": audio_set_of(r),
                    "stress": is_stress(r.get("sample_id", "")),
                    "wer_raw": float(r.get("wer_pct") or 0.0),
                    "cer_raw": float(r.get("cer_pct") or 0.0),
                    "wer_sem": compute_wer(ref_n, hyp_n) * 100,
                    "cer_sem": compute_cer(ref_n, hyp_n) * 100})
    return out


def summarize(rows: list[dict], include_stress: bool) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if not include_stress and r["stress"]:
            continue
        buckets[(r["model_id"], r["language"])].append(r)
    summary = []
    for (model, lang), rs in buckets.items():
        med = lambda key: round(statistics.median(x[key] for x in rs), 2)  # noqa: E731
        summary.append({"model_id": model, "language": lang,
                        "samples": len({x["sample_id"] for x in rs}), "runs": len(rs),
                        "wer_raw": med("wer_raw"), "wer_sem": med("wer_sem"),
                        "cer_raw": med("cer_raw"), "cer_sem": med("cer_sem"),
                        "wer_delta": round(med("wer_raw") - med("wer_sem"), 2)})
    return sorted(summary, key=lambda s: (s["language"], s["wer_sem"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-set", default="live", help="live | elevenlabs_rachel_mixed | ...")
    ap.add_argument("--language", default=None, help="ru | en; по умолчанию оба")
    ap.add_argument("--include-stress", action="store_true",
                    help="не исключать сэмплы с цифрами и именами")
    ap.add_argument("--show-deltas", action="store_true",
                    help="показать примеры, где нормализация изменила счёт сильнее всего")
    args = ap.parse_args()

    runs = json.loads(CONSOLIDATED.read_text())["runs_detail"]
    rows = [r for r in rescore(runs) if r["audio_set"] == args.audio_set]
    if args.language:
        rows = [r for r in rows if r["language"] == args.language]
    if not rows:
        raise SystemExit(f"нет записей для набора {args.audio_set!r}")

    summary = summarize(rows, args.include_stress)
    scope = "со стресс-сэмплами" if args.include_stress else "без стресс-сэмплов"
    print(f"Набор {args.audio_set}, {scope}, записей {len(rows)}\n")
    head = f"{'модель':<38} {'яз':<3} {'WER сырой':>10} {'WER сем.':>9} {'Δ':>7} {'CER сем.':>9}"
    print(head)
    print("-" * len(head))
    for s in summary:
        print(f"{s['model_id']:<38} {s['language']:<3} {s['wer_raw']:>9.1f}% "
              f"{s['wer_sem']:>8.1f}% {s['wer_delta']:>6.1f} {s['cer_sem']:>8.1f}%")

    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\n→ {OUT_CSV.relative_to(ROOT)}")

    if args.show_deltas:
        print("\nГде нормализация сняла больше всего (ошибка была форматированием):")
        worst = sorted(rows, key=lambda r: r["wer_sem"] - r["wer_raw"])[:5]
        for r in worst:
            print(f"\n  {r['model_id']} / {r['sample_id']}: "
                  f"{r['wer_raw']:.0f}% → {r['wer_sem']:.0f}%")
            print(f"    эталон:  {semantic_normalize(r['reference'])[:150]}")
            print(f"    гипотеза:{semantic_normalize(r['hypothesis'])[:150]}")


if __name__ == "__main__":
    main()
