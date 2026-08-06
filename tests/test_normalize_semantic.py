"""Семантическая нормализация: что она обязана схлопывать, а что обязана оставить
ошибкой. Граница ровно та же, что декларирует pipecat: формат чисел, произнесённая
вслух пунктуация и слова-паразиты не считаются ошибкой, а перепутанное имя или
число — считается."""
import pytest

from bench.normalize_semantic import semantic_normalize as sn


# --- формат числа не ошибка ---------------------------------------------------

@pytest.mark.parametrize("spoken,written", [
    ("двести один", "201"),
    ("восемьдесят три", "83"),
    ("тридцать восемь", "38"),
    ("сто два", "102"),
    ("six hundred", "600"),
    ("twenty", "20"),
    ("v three", "v3"),
    ("v one", "v1"),
])
def test_number_words_equal_digits(spoken, written):
    assert sn(spoken) == sn(written)


def test_ip_address_spoken_equals_written():
    """Главный источник дутого WER: эталон записан как звучало, модель пишет канонично."""
    assert sn("двести один точка сто два точка тридцать восемь точка семь") == sn("201.102.38.7")


def test_spoken_path_equals_written_path():
    assert sn("slash opt slash voice dash ai slash bot slash dot env") == sn("/opt/voice-ai/bot/.env")


def test_spoken_url_equals_written_url():
    assert sn("api dot groq dot com slash openai slash v one") == sn("api.groq.com/openai/v1")


def test_russian_spoken_symbols():
    assert sn("слэш opt слэш voice дэш ai") == sn("/opt/voice-ai")


def test_cyrillic_transliteration_stays_an_error():
    """«войс» вместо «voice» — это другой токен для агента, а не форматирование.
    Приравнивать алфавиты значило бы занижать WER."""
    assert sn("войс ай") != sn("voice ai")


# --- паразиты и пунктуация не ошибка ------------------------------------------

@pytest.mark.parametrize("noisy,clean", [
    ("um the model is fast", "the model is fast"),
    ("uh, so, like, it works", "so it works"),
    ("ну вот эээ модель быстрая", "модель быстрая"),   # затравка в начале реплики
    ("вот этот вариант", "вот этот вариант"),          # в середине «вот» осмысленно
    ("Модель быстрая!!!", "модель быстрая"),
])
def test_fillers_and_punctuation_ignored(noisy, clean):
    assert sn(noisy) == sn(clean)


def test_contractions_expanded():
    assert sn("it's not working") == sn("it is not working")


# --- а вот это обязано остаться ошибкой ---------------------------------------

def test_wrong_name_is_still_an_error():
    """pipecat считает ошибкой перепутанное имя — и мы тоже. Иначе метрика
    превращается в самообман."""
    assert sn("Шупилкин") != sn("Шпилкин")
    assert sn("smolevich voice bot") != sn("smolovich voice bot")


def test_wrong_number_is_still_an_error():
    assert sn("двести один") != sn("двести два")
    assert sn("six hundred") != sn("six thousand")


def test_negation_is_still_an_error():
    assert sn("модель не работает") != sn("модель работает")
    assert sn("it is not working") != sn("it is working")


# --- границы -------------------------------------------------------------------

def test_empty_and_none_safe():
    assert sn("") == ""
    assert sn(None) == ""


def test_spoken_symbol_converted_only_between_tokens():
    """«точка» превращается в разделитель, только если стоит МЕЖДУ двумя токенами.
    Иначе обычная фраза «в конце ставится точка» потеряет слово и WER занизится."""
    assert sn("сто точка два") == sn("100.2")
    assert "точка" in sn("в конце ставится точка")
    assert "dot" in sn("connect the dot")


def test_normalization_is_idempotent():
    """sn(sn(x)) == sn(x): иначе метрика зависит от того, сколько раз прогнали текст."""
    for s in ["двести один точка сто два", "uh so like it works", "v three dash turbo"]:
        assert sn(sn(s)) == sn(s)
