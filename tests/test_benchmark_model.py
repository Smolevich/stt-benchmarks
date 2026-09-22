"""Прогон модели по сэмплам: отказ провайдера на одном сэмпле не должен стирать остальные."""
from typing import Any

from bench.stt_landscape_bench import ModelSpec, RunnerCache, benchmark_model

SPEC = ModelSpec(id="fake-asr", runner="groq", model="fake-1", arch="AR", languages="2")


def sample(sample_id: str, language: str) -> dict[str, Any]:
    return {"id": sample_id, "language": language, "path": f"{sample_id}.mp3",
            "text": "привет мир", "duration_s": 1.0}


def transcribe_ru_only(path: str, language: str) -> str:
    if language != "ru":
        raise RuntimeError("language not supported")
    return "привет мир"


def run_mixed_languages() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    cache = RunnerCache()
    cache.loaded[SPEC.id] = transcribe_ru_only
    return benchmark_model(SPEC, [sample("ru-1", "ru"), sample("hy-1", "hy")],
                           cache, warmup=0, runs=1)


def test_unsupported_sample_does_not_drop_the_supported_ones():
    rows, _summary, _errors = run_mixed_languages()
    assert [row["sample_id"] for row in rows] == ["ru-1"]


def test_unsupported_sample_is_reported_as_error():
    _rows, _summary, errors = run_mixed_languages()
    assert errors == [{"model_id": "fake-asr", "sample_id": "hy-1", "error": "language not supported"}]
