"""Tests for the oruk-fourier-2b-mlx model registry entry."""

from bench.stt_landscape_bench import MODEL_REGISTRY


def test_oruk_fourier_2b_registered():
    assert "oruk-fourier-2b-mlx" in MODEL_REGISTRY
    spec = MODEL_REGISTRY["oruk-fourier-2b-mlx"]
    assert spec.runner == "mlx_audio_qwen3"
    assert spec.model == "oruk/oruk-fourier-2b"
    assert spec.languages == "EN"
