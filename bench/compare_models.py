"""Парное сравнение двух моделей на уже собранных транскриптах.

Ничего не запускает: берёт reference/hypothesis из results/consolidated.json.

    python3 -m bench.compare_models --a deepgram-nova-2 --b deepgram-nova-3 --language ru
    python3 -m bench.compare_models --a deepgram-nova-2 --b deepgram-nova-3 --language ru --raw
    python3 -m bench.compare_models --language ru --rank

Повторные прогоны одного файла схлопываются в одно наблюдение: у детерминированных
API они дают побайтово тот же текст (проверено — 79 из 80 пар), то есть меряют
разброс латентности, а не качества. Считать их отдельными наблюдениями значило бы
втрое занизить доверительный интервал.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from bench.normalize_semantic import semantic_normalize
from bench.stats import paired_bootstrap, pooled_wer, wilson, word_edits

ROOT = Path(__file__).resolve().parent.parent
CONSOLIDATED = ROOT / "results" / "consolidated.json"
STRESS = ("-04-digits", "-07-names")


def load(language: str, audio_set: str, include_stress: bool) -> tuple[dict, int]:
    """{model_id: {sample_id: (reference, hypothesis)}} + сколько прогонов разошлось."""
    runs = json.loads(CONSOLIDATED.read_text())["runs_detail"]
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        sid = r.get("sample_id", "")
        current_set = sid.split("__", 1)[1] if "__" in sid else "live"
        if current_set != audio_set or r.get("language") != language:
            continue
        if not include_stress and any(m in sid for m in STRESS):
            continue
        if not r.get("reference"):
            continue
        grouped[r["model_id"]][sid].append(r)

    out: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    disagreements = 0
    for model, samples in grouped.items():
        for sid, rs in samples.items():
            if len({x.get("hypothesis") or "" for x in rs}) > 1:
                disagreements += 1
            out[model][sid] = (rs[0]["reference"], rs[0].get("hypothesis") or "")
    return out, disagreements


def to_pairs(data: dict[str, tuple[str, str]], sample_ids: list[str], raw: bool) -> list:
    """Нормализуем один раз здесь, чтобы вся статистика считалась на одном тексте."""
    norm = (lambda s: (s or "").lower()) if raw else semantic_normalize
    return [(norm(data[s][0]), norm(data[s][1])) for s in sample_ids]


def empty_rate(data: dict[str, tuple[str, str]], sample_ids: list[str]) -> tuple[int, int]:
    """Пустая расшифровка при непустом эталоне — отдельный отказ, а не «WER 100%».
    Для агента это худший исход, и он обязан быть виден отдельной метрикой."""
    empties = sum(1 for s in sample_ids if data[s][0].strip() and not data[s][1].strip())
    return empties, len(sample_ids)


def report_pair(models: dict, a: str, b: str, raw: bool) -> None:
    for m in (a, b):
        if m not in models:
            raise SystemExit(f"нет данных для модели {m!r}. Есть: {', '.join(sorted(models))}")
    shared = sorted(set(models[a]) & set(models[b]))
    if not shared:
        raise SystemExit("у моделей нет общих сэмплов")

    pa, pb = to_pairs(models[a], shared, raw), to_pairs(models[b], shared, raw)
    res = paired_bootstrap(pa, pb)
    ka, na = empty_rate(models[a], shared)
    kb, _ = empty_rate(models[b], shared)

    metric = "сырой WER" if raw else "семантический WER"
    print(f"\n{a}  vs  {b}   ({metric}, общих реплик {len(shared)})\n")
    print(f"  pooled WER   {a:<26} {pooled_wer(pa):>6.1f}%")
    print(f"  pooled WER   {b:<26} {pooled_wer(pb):>6.1f}%")
    print(f"  разница (B минус A)                    {res['delta']:>+6.1f} пункта")
    print(f"  парный бутстрап 95% ДИ                 [{res['ci_low']:+.1f}; {res['ci_high']:+.1f}]")
    print(f"  p-value (двусторонний)                 {res['p_value']:.3f}")
    verdict = "разница значима" if res["significant"] else "РАЗНИЦА НЕ ЗНАЧИМА"
    print(f"  вывод                                  {verdict}")
    print(f"\n  разброс разностей по репликам sd       {res['per_sample_sd']:.1f}")
    need = res["required_n"]
    print(f"  реплик нужно для устойчивого вывода    "
          f"{'—' if need == float('inf') else int(need)}  (сейчас {len(shared)})")

    print("\n  пустые расшифровки при непустом эталоне (интервал Уилсона):")
    for model, k in ((a, ka), (b, kb)):
        lo, hi = wilson(k, na)
        print(f"    {model:<28} {k}/{na} = {k / na * 100:>5.1f}%   "
              f"[{lo * 100:.1f}%; {hi * 100:.1f}%]")

    worst = sorted(range(len(shared)),
                   key=lambda i: -( (word_edits(*pb[i])[0] / max(word_edits(*pb[i])[1], 1))
                                    - (word_edits(*pa[i])[0] / max(word_edits(*pa[i])[1], 1)))
                   )[:3]
    print("\n  реплики, где B отстаёт сильнее всего:")
    for i in worst:
        ea, wa = word_edits(*pa[i])
        eb, wb = word_edits(*pb[i])
        print(f"    {shared[i]:<22} A {ea / max(wa,1)*100:>5.1f}%  →  B {eb / max(wb,1)*100:>5.1f}%")


def report_rank(models: dict, raw: bool) -> None:
    rows = []
    for model, data in models.items():
        ids = sorted(data)
        k, n = empty_rate(data, ids)
        rows.append((pooled_wer(to_pairs(data, ids, raw)), model, len(ids), k, n))
    print(f"\n{'модель':<38}{'pooled WER':>11}{'реплик':>8}{'пустых':>8}")
    for wer, model, n, k, _ in sorted(rows):
        print(f"{model:<38}{wer:>10.1f}%{n:>8}{k:>8}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="базовая модель")
    ap.add_argument("--b", help="сравниваемая модель")
    ap.add_argument("--language", default="ru", choices=("ru", "en"))
    ap.add_argument("--audio-set", default="live")
    ap.add_argument("--include-stress", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="считать по сырому тексту, без семантической нормализации")
    ap.add_argument("--rank", action="store_true", help="таблица всех моделей по pooled WER")
    args = ap.parse_args()

    models, disagreements = load(args.language, args.audio_set, args.include_stress)
    if not models:
        raise SystemExit("нет подходящих записей")
    if disagreements:
        print(f"⚠ прогонов с расходящимся текстом: {disagreements} — берётся первый")

    if args.rank or not (args.a and args.b):
        report_rank(models, args.raw)
        if not (args.a and args.b):
            return
    report_pair(models, args.a, args.b, args.raw)


if __name__ == "__main__":
    main()
