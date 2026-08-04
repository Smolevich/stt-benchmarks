# Architecture

- **`stt_landscape_bench.py`** is the single entry point. `MODEL_REGISTRY` maps model IDs → `ModelSpec(runner, model, arch, languages, ...)`. The `runner` field selects a branch in `build_runner()` which dynamically imports the heavy framework (mlx_whisper / nemo / transformers / parakeet_mlx / gigaam / funasr / httpx for cloud APIs). Imports are deferred so you only need the deps for the models you actually run.
- **`RunnerCache`** keeps each loaded model in memory across samples within a single bench invocation (important: NeMo and transformers loads are slow).
- **`PeakMemorySampler`** is a background thread that samples `psutil` RSS for the process + children every 20 ms while a transcription runs.
- **Latency metric** is `elapsed_s / audio_duration_s * 10` — seconds of wall time per 10 seconds of audio. Audio duration comes from `ffprobe`, so ffmpeg must be on PATH.
- **Cloud runners** (`groq`, `elevenlabs`, `fish_audio`, `deepgram`) gate on `requires_env` and accept `--cloud-sleep-s` to dodge rate limits. All post multipart form data except `deepgram`, which takes raw audio bytes as the request body and every option as a query parameter — see [deepgram.md](deepgram.md).
- **`language_override`** on `ModelSpec` pins a fixed language code for every sample (empty for all models except `deepgram-nova-3-multi`, which always sends `language=multi`). Everything else takes the per-sample code from the manifest.
- **`wer_eval.py`** is a self-contained Levenshtein implementation (no `jiwer` dep). Normalization lowercases, replaces `-` with space, and strips non-word characters before splitting. CER works on raw lowercased characters with no normalization.

## Adding a model

1. Add a `ModelSpec` entry to `MODEL_REGISTRY` in `bench/stt_landscape_bench.py`.
2. If it needs a new `runner`, add a branch to `build_runner()` returning a `transcribe(path, language) -> str` callable. Keep heavy imports inside the branch.
3. For cloud APIs, set `requires_env` so `build_runner` fails fast with a clear error when the key is missing.
