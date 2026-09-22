# Commands

```bash
# List the model registry
python3 bench/stt_landscape_bench.py --list-models

# Smoke test against one MLX model
python3 bench/stt_landscape_bench.py \
    --samples samples/manifest.example.json \
    --out-dir results/ \
    --models mlx-whisper-small \
    --warmup 1 --runs 3

# Full RU + EN sweep with cloud models (needs API keys in env)
GROQ_API_KEY=... ELEVENLABS_API_KEY=... FISH_AUDIO_API_KEY=... DEEPGRAM_API_KEY=... \
python3 bench/stt_landscape_bench.py \
    --samples samples/manifest.json \
    --out-dir results/ \
    --models all \
    --cloud-sleep-s 1.0

# Merge all per-run JSONs in results/ into consolidated.json + summary CSV
python3 bench/consolidate_results.py --results-dir results/

# Quick standalone WER/CER on two strings
python3 bench/wer_eval.py "reference text" "hypothesis text"

# Render plots
python3 bench/plot_ar_vs_nar.py --out plots/ar_vs_nar.png
python3 bench/plot_stt_landscape.py results/summary.csv --language all --out plots/stt_latency_wer.png
```

`consolidate_results.py` globs `stt_landscape_2026*.json` — bench output filenames must keep the `stt_landscape_<UTC timestamp>.json` format for it to pick them up. It dedups by `(model_id, sample_id, run_index)` and the later file (sorted by name = timestamp) wins, so re-running a model replaces earlier results without manual cleanup. Edge-TTS synthetic samples (`ru-dmitry`, `ru-svetlana`, `en-aria`) are excluded by default; pass `--include-edge-tts` to keep them.
