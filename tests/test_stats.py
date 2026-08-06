"""Статистика сравнения моделей: pooled WER, интервал Уилсона, парный бутстрап."""
import pytest

from bench.stats import paired_bootstrap, pooled_wer, required_samples, wilson, word_edits


# --- pooled WER ---------------------------------------------------------------

def test_word_edits_counts_substitution_insertion_deletion():
    assert word_edits("a b c", "a b c") == (0, 3)
    assert word_edits("a b c", "a x c") == (1, 3)   # замена
    assert word_edits("a b c", "a b") == (1, 3)     # пропуск
    assert word_edits("a b c", "a b c d") == (1, 3)  # вставка


def test_pooled_is_not_the_mean_of_per_sample_wer():
    """Главная причина заводить pooled: длинная реплика должна весить больше
    короткой. Средняя по сэмплам этого не делает и прячет хвосты."""
    pairs = [("a b c d e f g h i j", "a b c d e f g h i j"),  # 10 слов, 0 ошибок
             ("x", "y")]                                       # 1 слово, 1 ошибка
    assert pooled_wer(pairs) == pytest.approx(1 / 11 * 100)
    # среднее по сэмплам дало бы 50% — то есть в пять раз больше


def test_pooled_catches_empty_transcript():
    """Пустая расшифровка = все слова потеряны. Медиана такое почти не заметит,
    pooled обязан."""
    pairs = [("a b c d", "a b c d")] * 9 + [("a b c d", "")]
    assert pooled_wer(pairs) == pytest.approx(10.0)


def test_pooled_empty_reference_is_skipped():
    assert pooled_wer([("", "что-то"), ("a b", "a b")]) == 0.0


# --- интервал Уилсона ---------------------------------------------------------

def test_wilson_matches_known_values():
    lo, hi = wilson(1, 10)
    assert lo == pytest.approx(0.018, abs=0.002)
    assert hi == pytest.approx(0.404, abs=0.002)


def test_wilson_narrows_with_more_samples():
    _, hi_small = wilson(1, 10)
    _, hi_big = wilson(10, 100)
    assert hi_big < hi_small


def test_wilson_handles_zero_and_full():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and 0 < lo < 1


# --- парный бутстрап ----------------------------------------------------------

def _pairs(n, wer_pct):
    """n реплик по 10 слов, в каждой ровно wer_pct/10 ошибок."""
    bad = int(round(wer_pct / 10))
    ref = " ".join(f"w{i}" for i in range(10))
    hyp = " ".join(("x" if i < bad else f"w{i}") for i in range(10))
    return [(ref, hyp)] * n


def test_identical_models_give_interval_containing_zero():
    a = _pairs(20, 20.0)
    res = paired_bootstrap(a, list(a), seed=1)
    assert res["delta"] == pytest.approx(0.0)
    assert res["ci_low"] <= 0 <= res["ci_high"]
    assert not res["significant"]


def test_clearly_worse_model_is_detected():
    a, b = _pairs(20, 10.0), _pairs(20, 50.0)
    res = paired_bootstrap(a, b, seed=1)
    assert res["delta"] == pytest.approx(40.0, abs=1.0)
    assert res["ci_low"] > 0 and res["significant"]


def test_paired_uses_sample_alignment():
    """Парность в том, что обе модели слушали одно аудио: если одна всегда хуже
    ровно на 10 пунктов, разброс разностей нулевой и вывод уверенный — даже когда
    сами сэмплы очень разные по сложности."""
    a = _pairs(5, 10.0) + _pairs(5, 60.0)
    b = _pairs(5, 20.0) + _pairs(5, 70.0)
    res = paired_bootstrap(a, b, seed=1)
    assert res["ci_low"] > 0 and res["ci_high"] < 20


def test_bootstrap_is_deterministic_for_a_seed():
    a, b = _pairs(10, 10.0), _pairs(10, 30.0)
    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap(_pairs(3, 10.0), _pairs(4, 10.0), seed=1)


# --- сколько сэмплов нужно ----------------------------------------------------

def test_required_samples_grows_with_noise_and_shrinks_with_effect():
    assert required_samples(effect=5.0, sd=10.0) < required_samples(effect=5.0, sd=30.0)
    assert required_samples(effect=20.0, sd=10.0) < required_samples(effect=2.0, sd=10.0)


def test_required_samples_infinite_when_no_effect():
    assert required_samples(effect=0.0, sd=10.0) == float("inf")
