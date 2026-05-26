# Python environments

Two venvs because GigaAM pins `torch<=2.5.1`, which has no wheels for Python 3.14 (and is unhappy on 3.13). The README walks a user through the same split.

- **`.venv` — Python 3.12+** for `mlx-whisper` and cloud runners. Install: `pip install -r requirements.txt`. Covers `mlx-whisper-*`, `groq-*`, `elevenlabs-*`.
- **`.venv-nar` — Python 3.11 required** for NAR/heavyweight runners. Install: `pip install -r requirements-nar.txt`. Covers `gigaam-*`, `parakeet-tdt-*` (both `transformers` and `parakeet-mlx`), `transformers-moonshine-*`, `sensevoice-*`.

NeMo runners (`nemo-*`) need `pip install 'nemo_toolkit[asr]'` on top — not in either requirements file because the install is heavy and most users don't need NeMo-native variants (the `transformers` and `parakeet-mlx` Parakeet runners cover the same weights).

`ffmpeg` / `ffprobe` must be on PATH — used by `audio_duration_s()` to compute the latency-per-10s metric.
