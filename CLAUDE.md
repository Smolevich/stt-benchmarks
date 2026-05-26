# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Reproducible STT benchmark harness for the "STT landscape" article. Compares local (MLX / NeMo / transformers / parakeet-mlx / GigaAM / funasr) and cloud (Groq, ElevenLabs) speech-to-text models on RU + EN samples. Outputs WER/CER, latency-per-10s-audio, and peak RSS.

[README.md](README.md) is the user-facing entry point — TL;DR, result tables, reproduction steps, methodology, limitations. Read it first; this file is the orientation map for working *on* the repo.

## Layout

- `bench/` — Python scripts (benchmark runner, WER/CER, consolidation, plots).
- `samples/` — `recording-script.md`, `manifest.template.json`, `manifest.example.json` (Edge-TTS smoke set). `manifest.json` and `audio/` are gitignored (personal voice = PII).
- `results/` — committed `consolidated.json` + `summary.csv`. Per-run `*_2026*.json/csv` are gitignored.
- `plots/` — rendered figures.
- `docs/` — detailed working notes.
- `requirements.txt` / `requirements-nar.txt` — two separate envs (see below).

## Detailed docs

- [docs/commands.md](docs/commands.md) — bench / consolidate / plot invocations.
- [docs/environments.md](docs/environments.md) — why there are two venvs (Python 3.12 vs 3.11) and which models live where.
- [docs/architecture.md](docs/architecture.md) — runner registry, caches, latency metric, how to add a model.
- [docs/path-caveat.md](docs/path-caveat.md) — **read before running**: `from experiments.wer_eval import ...` in `stt_landscape_bench.py` is broken and needs patching on a fresh clone.
