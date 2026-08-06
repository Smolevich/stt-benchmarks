"""Статистические примитивы для сравнения моделей.

Чистые функции без ввода-вывода — их и тестируем; чтение результатов живёт в
bench/compare_models.py.

Три вещи, которых не хватало бенчу:

1. **pooled WER** — сумма ошибок делить на сумму слов эталона. Стандарт в ASR
   (pipecat отдаёт его отдельной колонкой). Медиана по сэмплам, которой бенч
   пользовался, прячет хвост: пустая расшифровка одного сэмпла её почти не двигает,
   хотя для голосового агента это худшее, что может случиться.
2. **интервал Уилсона** — для доли (пустые ответы, отказы). Для WER он неприменим:
   там не «успех из испытаний», а расстояние редактирования к длине эталона, ошибки
   внутри реплики скоррелированы, а длины разные.
3. **парный бутстрап** — для «модель A лучше модели B». Парный, потому что обе
   слушали одно и то же аудио: сравниваем разности по сэмплам, и разброс «этот
   сэмпл труден для всех» уходит из оценки.
"""
from __future__ import annotations

import math
import random

Pair = tuple[str, str]  # (эталон, гипотеза), уже нормализованные


def word_edits(reference: str, hypothesis: str) -> tuple[int, int]:
    """(расстояние редактирования в словах, длина эталона)."""
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0, 0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(prev[j - 1] if r == h else 1 + min(prev[j - 1], prev[j], cur[j - 1]))
        prev = cur
    return prev[-1], len(ref)


def pooled_wer(pairs: list[Pair]) -> float:
    """Сумма ошибок / сумма слов эталона, в процентах. Реплики с пустым эталоном
    пропускаем — делить не на что."""
    total_e = total_w = 0
    for ref, hyp in pairs:
        e, w = word_edits(ref, hyp)
        total_e += e
        total_w += w
    return (total_e / total_w * 100) if total_w else 0.0


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Доверительный интервал Уилсона для доли. Для долей — в отличие от Вальда —
    ведёт себя прилично на краях и на малых n, а это ровно наш случай: пустых
    ответов единицы."""
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_bootstrap(a: list[Pair], b: list[Pair], *, n_boot: int = 20000,
                     seed: int = 42, z: float = 1.96) -> dict:
    """Насколько pooled WER модели B выше, чем у A, и можно ли этому верить.

    a[i] и b[i] — одна и та же реплика, услышанная двумя моделями; порядок обязан
    совпадать. На каждой итерации пересэмплируем НОМЕРА реплик и пересчитываем
    pooled по обеим моделям на одном и том же наборе — в этом и парность.
    """
    if len(a) != len(b):
        raise ValueError("парное сравнение требует одинакового набора реплик")
    if not a:
        raise ValueError("нечего сравнивать")

    ea = [word_edits(r, h) for r, h in a]
    eb = [word_edits(r, h) for r, h in b]
    delta = pooled_wer(b) - pooled_wer(a)

    rnd = random.Random(seed)
    n = len(a)
    deltas = []
    for _ in range(n_boot):
        idx = [rnd.randrange(n) for _ in range(n)]
        wa = sum(ea[i][1] for i in idx)
        wb = sum(eb[i][1] for i in idx)
        if not wa or not wb:
            continue
        deltas.append(sum(eb[i][0] for i in idx) / wb * 100
                      - sum(ea[i][0] for i in idx) / wa * 100)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(int(0.975 * len(deltas)), len(deltas) - 1)]

    # Достигнутый уровень значимости: как часто разность меняет знак. Умножаем на 2 —
    # гипотеза двусторонняя, нам неважно, в какую сторону ошибиться.
    worse = sum(1 for d in deltas if d <= 0) / len(deltas)
    p_value = min(1.0, 2 * min(worse, 1 - worse))

    # Разброс разностей ПО СЭМПЛАМ — основа оценки нужного размера выборки.
    per_sample = [(eb[i][0] / eb[i][1] * 100 if eb[i][1] else 0.0)
                  - (ea[i][0] / ea[i][1] * 100 if ea[i][1] else 0.0)
                  for i in range(n)]
    mean_d = sum(per_sample) / n
    sd = math.sqrt(sum((x - mean_d) ** 2 for x in per_sample) / (n - 1)) if n > 1 else 0.0

    return {"n": n, "delta": delta, "ci_low": lo, "ci_high": hi, "p_value": p_value,
            "significant": lo > 0 or hi < 0,
            "per_sample_mean": mean_d, "per_sample_sd": sd,
            "required_n": required_samples(mean_d, sd)}


def required_samples(effect: float, sd: float, z: float = 1.96,
                     z_power: float = 0.84) -> float:
    """Сколько РАЗНЫХ реплик нужно, чтобы эффект такого размера был устойчив
    (95% значимость, 80% мощность). Повторные прогоны одного файла сюда не идут:
    у детерминированного API они дают тот же текст, то есть одно наблюдение."""
    if not effect:
        return float("inf")
    return math.ceil(((z + z_power) * sd / abs(effect)) ** 2)
