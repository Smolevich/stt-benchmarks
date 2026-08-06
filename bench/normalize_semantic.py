"""Семантическая нормализация текста перед подсчётом WER.

Зачем: в этом бенче эталон записан так, КАК ПРОИЗНОСИЛОСЬ («двести один точка сто
два», «slash opt slash voice dash ai»), а модели пишут канонично («201.102.38.7»,
«/opt/voice-ai»). Обычный WER считает это ошибкой и раздувается до 60-95% на
сэмплах с числами и путями — при том, что для голосового агента обе записи значат
ровно одно и то же.

Граница проведена там же, где её проводит pipecat в своём semantic WER: формат
чисел, произнесённая вслух пунктуация, стяжения и слова-паразиты ошибкой НЕ
считаются; перепутанное имя, другое число и потерянное отрицание — считаются.
Отличие от pipecat: там судит LLM, здесь детерминированные правила — дешевле,
воспроизводимо и не зависит от настроения модели-судьи.

Чего сознательно НЕ делаем: не приравниваем кириллическую транслитерацию к
латинице («войс» ≠ «voice»). Модель, записавшая технический термин не тем
алфавитом, выдала агенту другой токен — это ошибка, а не форматирование.

Порядок шагов важен: числительные схлопываются ДО удаления произнесённых
символов, иначе «сто точка два» слипается в «102» вместо «100 2».
"""
from __future__ import annotations

import re

# --- слова-паразиты ----------------------------------------------------------
_FILLERS = {
    "um", "uh", "erm", "hmm", "mmm", "like", "basically", "actually",
    "эм", "ээ", "эээ", "ммм", "мм", "ааа", "типа", "короче",
}
# «ну» и «вот» осмысленны в середине фразы («ну и что», «вот этот»), поэтому
# вычищаются только в начале реплики, где они гарантированно затравка.
_LEADING_FILLERS = {"ну", "вот", "so", "well"}

# --- стяжения ----------------------------------------------------------------
_CONTRACTIONS = {
    "it's": "it is", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "can not", "cannot": "can not", "won't": "will not", "wouldn't": "would not",
    "shouldn't": "should not", "couldn't": "could not", "i'm": "i am", "you're": "you are",
    "we're": "we are", "they're": "they are", "i've": "i have", "you've": "you have",
    "we've": "we have", "they've": "they have", "i'll": "i will", "you'll": "you will",
    "it'll": "it will", "that's": "that is", "there's": "there is", "here's": "here is",
    "what's": "what is", "let's": "let us", "hasn't": "has not", "haven't": "have not",
}

# --- произнесённые вслух символы ---------------------------------------------
# Все схлопываются в НИЧТО: различать «.» и «/» смысла нет — «api dot groq dot com
# slash openai» и «api.groq.com/openai» это один и тот же адрес.
_SPOKEN_SYMBOLS = {
    "dot", "slash", "dash", "underscore", "hyphen", "colon", "backslash", "point",
    "точка", "слэш", "слеш", "дэш", "деш", "тире", "дефис", "подчеркивание",
    "двоеточие",
}
_SYMBOL_CHARS = ".,/\\-_:;!?()[]{}\"«»…"

# --- числительные ------------------------------------------------------------
_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
          "eighteen": 18, "nineteen": 19,
          "ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
          "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
          "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
          "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
          "девятнадцать": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90,
         "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
         "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90}
_HUNDREDS = {"сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500,
             "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9,
           "тысяча": 1000, "тысячи": 1000, "тысяч": 1000,
           "миллион": 10 ** 6, "миллиона": 10 ** 6, "миллионов": 10 ** 6}

_SPLIT_ALNUM_RE = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])", re.UNICODE)
_WORDISH_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Строка → токены. Символы-разделители становятся пробелами («voice-ai» →
    «voice ai», «201.102» → «201 102»), а стык буквы и цифры разрезается («v3» →
    «v 3»), чтобы письменная форма совпала с произнесённой («v three»)."""
    text = re.sub(rf"[{re.escape(_SYMBOL_CHARS)}]+", " ", text)
    out: list[str] = []
    for raw in text.split():
        for piece in _WORDISH_RE.findall(raw):
            out.extend(p for p in _SPLIT_ALNUM_RE.split(piece) if p)
    return out


def _expand_contractions(text: str) -> str:
    def sub(m: re.Match) -> str:
        return _CONTRACTIONS.get(m.group(0), m.group(0))
    return re.sub(r"[a-z]+'[a-z]+", sub, text)


def _read_number(tokens: list[str], start: int) -> tuple[int, int]:
    """(значение, сколько токенов съедено). (0, 0) — числа здесь нет."""
    total, current, consumed, seen = 0, 0, 0, False
    i = start
    while i < len(tokens):
        t = tokens[i]
        if t in _UNITS:
            current += _UNITS[t]
        elif t in _TENS:
            current += _TENS[t]
        elif t in _HUNDREDS:
            current += _HUNDREDS[t]
        elif t in _SCALES:
            scale = _SCALES[t]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
        else:
            break
        seen = True
        consumed += 1
        i += 1
    return (total + current, consumed) if seen else (0, 0)


def _collapse_numbers(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        value, consumed = _read_number(tokens, i)
        if consumed:
            out.append(str(value))
            i += consumed
        else:
            out.append(tokens[i])
            i += 1
    return out


def _drop_spoken_symbols(tokens: list[str]) -> list[str]:
    """Произнесённый символ выкидываем везде, КРОМЕ последнего токена: «в конце
    ставится точка» — это существительное, а не разделитель, и терять его нельзя."""
    last = len(tokens) - 1
    return [t for i, t in enumerate(tokens) if not (t in _SPOKEN_SYMBOLS and i != last)]


def _drop_fillers(tokens: list[str]) -> list[str]:
    """Сначала убираем безусловные паразиты, и только ПОТОМ смотрим на начало
    реплики. Иначе «uh so like it works» и «so it works» нормализуются по-разному:
    в первом «so» стоит не первым и уцелеет, во втором — исчезнет."""
    tokens = [t for t in tokens if t not in _FILLERS]
    i = 0
    while i < len(tokens) and tokens[i] in _LEADING_FILLERS:
        i += 1
    return tokens[i:]


def semantic_normalize(text: str | None) -> str:
    """Текст → канонический вид для сравнения. Пустой вход даёт пустую строку."""
    if not text:
        return ""
    lowered = re.sub(r"[’`]", "'", text.lower()).replace("ё", "е")
    tokens = _tokenize(_expand_contractions(lowered))
    tokens = _collapse_numbers(tokens)
    tokens = _drop_spoken_symbols(tokens)
    tokens = _drop_fillers(tokens)
    return " ".join(tokens)
