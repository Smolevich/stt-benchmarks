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

# Synthetic noise set: regenerate the noisy variants from the committed clean masters
python3 bench/mix_noise.py --list
python3 bench/mix_noise.py

# Synthetic noise sweep (RU + HY, clean / moderate / heavy)
GROQ_API_KEY=... python3 bench/stt_landscape_bench.py \
    --samples samples/manifest_synthetic_noise.json \
    --out-dir results/ \
    --models groq-whisper-large-v3-turbo,groq-whisper-large-v3 \
    --cloud-sleep-s 4 --warmup 1 --runs 3

# Merge all per-run JSONs in results/ into consolidated.json + summary CSV
python3 bench/consolidate_results.py --results-dir results/

# Quick standalone WER/CER on two strings
python3 bench/wer_eval.py "reference text" "hypothesis text"

# Render plots
python3 bench/plot_ar_vs_nar.py --out plots/ar_vs_nar.png
python3 bench/plot_stt_landscape.py results/summary.csv --language all --out plots/stt_latency_wer.png
```

`mix_noise.py` rebuilds `samples/synthetic/elevenlabs_sarah_noise/*-noise.mp3` and `*-heavy.mp3` from the committed clean TTS masters in the same directory. Noise is a pipeline step, not a property of the file: the recipe (SNR, ring tone, padding, RNG seed) lives in `CASES` inside the script, so a new noise level costs an ffmpeg run, not a new TTS render. The seed is fixed — without it every regeneration would move WER.

`consolidate_results.py` globs `stt_landscape_2026*.json` — bench output filenames must keep the `stt_landscape_<UTC timestamp>.json` format for it to pick them up. It dedups by `(model_id, sample_id, run_index)` and the later file (sorted by name = timestamp) wins, so re-running a model replaces earlier results without manual cleanup. Edge-TTS synthetic samples (`ru-dmitry`, `ru-svetlana`, `en-aria`) are excluded by default; pass `--include-edge-tts` to keep them.
