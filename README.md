# STT Benchmarks — 12 Models on Live Voice, RU + EN

Hands-on benchmark of 12 speech-to-text models on 20 live voice memos (10 RU + 10 EN), recorded on iPhone Voice Memos and run on a Mac mini M4 Pro. Comparing local (mlx-whisper, GigaAM, Parakeet, Moonshine, Sense Voice) against cloud (Groq Whisper Turbo, ElevenLabs Scribe v1, ElevenLabs Scribe v1 experimental).

Companion to the LinkedIn carousel and blog post — link added once published.

## TL;DR

**ElevenLabs Scribe v1 experimental is the strongest model in both languages.** 11.5% WER on RU and 11.5% WER on EN (clean subset). Same provider's `scribe_v1` (the older default) gives 23.1% on RU — two-times worse — so the API parameter alone makes the biggest difference. Architecture (AR vs NAR), specialization, runtime, and cloud-vs-local all matter less than the model version.

See `results/summary.csv` for full ranking; `results/consolidated.json` for raw transcripts.

## What's inside

```
bench/
  stt_landscape_bench.py     # main bench runner (12 models, multiple runners)
  wer_eval.py                 # WER/CER calculator
  consolidate_results.py      # merge multiple bench runs into one summary
  plot_ar_vs_nar.py           # generates the AR vs NAR explainer diagram
  plot_stt_landscape.py       # latency × WER scatter plot from CSV
samples/
  recording-script.md         # what to record — 20 scenarios with full text
  manifest.template.json      # reference text for 20 samples, paths to fill in
  manifest.example.json       # example manifest with edge-tts smoke-test samples
  manifest_synthetic.json     # committed TTS-generated RU + EN synthetic set
  manifest_silero_*.json      # committed independent RU synthetic control sets
results/
  consolidated.json           # all runs deduplicated, with transcripts
  summary.csv                 # medians by (model, language)
  synthetic_consolidated.json # latest curated synthetic benchmark snapshot
  synthetic_summary.csv       # latest curated synthetic summary
plots/
  ar_vs_nar.png               # generated illustration of AR vs NAR concept
```

Live audio files are **not included** — they're personal voice recordings. TTS-generated synthetic audio is committed separately under `samples/synthetic/` because it contains no personal voice data. To reproduce the primary benchmark on your own voice, see "Reproducing the bench" below.

## Final numbers (8 sample × 3 runs, WER clean)

WER clean excludes `*-04-digits` and `*-07-names` — those two samples push WER to 70-95% across all models because Whisper normalizes "twenty" → "20" and doesn't know identifiers like `smolevich_voice_bot`. The artifact is real but masks model differences.

### Russian — 11 models tested

| # | Model | Where | Latency / 10s | WER | CER |
|---|---|---|---|---|---|
| 1 | ElevenLabs Scribe v1 experimental | cloud | 0.82 s | **11.5%** | 10.1% |
| 2 | Groq Whisper L-v3 Turbo | cloud | 0.21 s | 17.9% | 15.6% |
| 3 | GigaAM v2 CTC | local | 0.42 s | 18.2% | 17.8% |
| 4 | mlx-whisper-large-v3-turbo | local | 0.69 s | 19.4% | 17.3% |
| 5 | mlx-whisper-medium | local | 0.96 s | 20.0% | 13.1% |
| 6 | GigaAM v2 RNN-T | local | 0.49 s | 21.0% | 18.9% |
| 7 | Parakeet TDT v3 (mlx) | local | 0.22 s | 21.0% | 15.2% |
| 8 | ElevenLabs Scribe v1 (legacy) | cloud | 0.93 s | 23.1% | 22.0% |
| 9 | mlx-whisper-large-v3 | local | 1.34 s | 24.3% | 16.3% |
| 10 | T-one | local | 3.16 s | 28.5% | 20.8% |
| 11 | mlx-whisper-small | local | 0.52 s | 28.8% | 17.6% |

### English — 10 models tested

| # | Model | Where | Latency / 10s | WER | CER |
|---|---|---|---|---|---|
| 1 | ElevenLabs Scribe v1 experimental | cloud | 0.63 s | **11.5%** | 7.8% |
| 2 | ElevenLabs Scribe v1 | cloud | 0.62 s | 12.7% | 8.4% |
| 3 | Groq Whisper L-v3 Turbo | cloud | 0.16 s | 15.0% | 7.8% |
| 4 | mlx-whisper-large-v3-turbo | local | 0.52 s | 15.5% | 8.1% |
| 5 | mlx-whisper-large-v3 | local | 0.97 s | 17.1% | 8.7% |
| 6 | mlx-whisper-medium | local | 0.72 s | 17.7% | 7.4% |
| 7 | Parakeet TDT v3 (mlx) | local | 0.18 s | 17.9% | 10.0% |
| 8 | mlx-whisper-small | local | 0.41 s | 22.4% | 11.5% |
| 9 | Moonshine base | local | 2.26 s | 24.7% | 14.2% |
| 10 | Sense Voice small | local | 1.38 s | 31.7% | 17.4% |

`mlx` after Parakeet means it ran via the [`parakeet-mlx`](https://github.com/senstella/parakeet-mlx) community port. Through `transformers` (PyTorch CPU) the same model gives 23.0% RU / 21.7% EN at 1.15-1.48 s latency — see "Runtime matters" below.

## What I found

1. **Model version is the biggest delta.** `scribe_v1` → `scribe_v1_experimental` cuts RU WER from 23.1% to 11.5% — same provider, same API, different `model_id`. If you use ElevenLabs Scribe in production and never re-checked which version, you may be sitting on 2× the error rate.

2. **Runtime is the second-biggest delta on Apple Silicon.** Parakeet TDT v3 through generic `transformers.pipeline` (PyTorch CPU) → 1.48 s, 21.7% WER (EN). Same weights through `parakeet-mlx` (Apple Silicon native) → 0.18 s, 17.9% WER. 8× faster, 3.8 points more accurate. If you're benchmarking NAR models on a Mac without their native runtime, your numbers are wrong.

3. **Specialization wins by less than it first appears.** GigaAM (RU-only, Sber) at 18.2% beat Parakeet (multilingual NAR, NVIDIA) at 21.0% on RU — 2.8 points gap. Earlier (before MLX) Parakeet was at 23.0% and the gap was 4.8 points; runtime hid part of it.

4. **Cloud vs local is mostly noise on quality.** Best local model (mlx-whisper-large-v3-turbo) gives 19.4% RU / 15.5% EN. Best cloud (Groq Whisper Turbo) gives 17.9% / 15.0%. Difference of 1.5 points RU and 0.5 EN — within run-to-run variance on a small sample. **Latency** is the real cloud advantage: Groq is fastest by a wide margin thanks to LPU hardware.

5. **Parakeet TDT v3 is competitive on EN — but only with the right runtime.** With MLX, it sits right next to Whisper-medium (17.9% vs 17.7%) at one-quarter the latency (0.18 s vs 0.72 s). It's a viable local NAR option on Apple Silicon.

6. **Multilingual NAR models lose to specialized models on a language they aren't tuned for.** Sense Voice (Alibaba, 50+ languages) gave 31.7% WER on English — worst in the set. Specialization clearly matters.

## Reproducing the bench

You need a Mac (Apple Silicon recommended for mlx-whisper) or any Linux machine for the non-MLX models.

### 1. Install

```bash
git clone https://github.com/<your-handle>/stt-benchmarks
cd stt-benchmarks

# Main env (mlx-whisper)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For NAR models (GigaAM, Parakeet, Moonshine, Sense Voice) — needs Python 3.11 + torch 2.5
python3.11 -m venv .venv-nar
source .venv-nar/bin/activate
pip install -r requirements-nar.txt
```

### 2. Record samples

Open `samples/recording-script.md`. It lists 20 scenarios with the text for each. Record on iPhone Voice Memos (or any recorder), save as `.m4a`/`.mp3`/`.wav`, name them `ru-01-clean.m4a` ... `en-10-podcast.m4a`, and put them in `samples/audio/`.

Sanity check: each file should have mean volume ≥ -70 dB. Silent recordings give 100% WER and pollute results.

```bash
for f in samples/audio/*.m4a; do
  ffmpeg -i "$f" -af "volumedetect" -f null /dev/null 2>&1 | grep mean_volume
done
```

### 3. Update manifest

Copy `samples/manifest.template.json` to `samples/manifest.json`. Verify each `path` points to your recorded file.

### 4. Run

```bash
# Local mlx-whisper (Apple Silicon)
python bench/stt_landscape_bench.py \
  --samples samples/manifest.json \
  --models mlx-whisper-small,mlx-whisper-medium,mlx-whisper-large-v3-turbo,mlx-whisper-large-v3 \
  --warmup 1 --runs 3

# Cloud (set keys in env)
GROQ_API_KEY=... ELEVENLABS_API_KEY=... python bench/stt_landscape_bench.py \
  --samples samples/manifest.json \
  --models groq-whisper-large-v3-turbo,elevenlabs-scribe-v1-experimental,elevenlabs-scribe-v1 \
  --cloud-sleep-s 4 --warmup 1 --runs 3

# NAR models (use venv with torch 2.5)
.venv-nar/bin/python bench/stt_landscape_bench.py \
  --samples samples/manifest.json \
  --models gigaam-v2-ctc,gigaam-v2-rnnt,parakeet-tdt-0.6b-v3-mlx,parakeet-tdt-0.6b-v3,sensevoice-small,transformers-moonshine-base,t-one \
  --warmup 1 --runs 3
```

### 5. Consolidate

```bash
python bench/consolidate_results.py
```

Produces `results/consolidated_<timestamp>.json` (all runs deduped) and `consolidated_summary_<timestamp>.csv` (medians).

## Methodology

- **Hardware**: Mac mini M4 Pro, 64 GB RAM, no discrete GPU.
- **Primary audio**: 20 voice memos recorded on iPhone (one speaker, m4a, 48 kHz mono, ~10-25 s each). **This is the most unbiased dataset** — real microphone, natural prosody, genuine speaker habits. Live voice numbers are the primary ranking signal.
- **Synthetic audio**: secondary TTS-generated control sets (`elevenlabs_rachel_mixed` via ElevenLabs Rachel, and Silero RU voices). Added for reproducibility (live audio is not committed — it's personal voice / PII) and to illustrate how TTS-generated audio can shift model rankings due to clean audio characteristics and potential vendor self-bias. **Do not treat synthetic results as the final benchmark.** See [docs/methodology.md](docs/methodology.md) for a detailed explanation.
- **Scenarios**: clean room, fast speech, whisper, street noise, numbers spoken as words, proper names/identifiers, code-switching RU↔EN, podcast pace, short commands, long sentence with subordinate clauses.
- **Metric**: WER and CER via Levenshtein distance on lowercased, punctuation-stripped text. No advanced normalization (numerals vs words, etc.) — see "Limitations".
- **Reps**: 1 warmup + 3 measured runs per (model, sample), report medians.
- **Cloud rate limits**: 4 sec sleep between cloud requests to dodge Groq free-tier limits.
- **WER clean**: subset excluding `*-04-digits` and `*-07-names` to remove the dominant noise of word-vs-digit normalization mismatch.

## Limitations

- **Single speaker, single recording device.** Results are biased toward the bench operator's voice and an iPhone's mic. Other speakers, accents, and devices will give different numbers.
- **Small sample size.** 10 sample × 3 runs per language. Confidence intervals are wide — treat differences smaller than ~2 points WER as noise.
- **Synthetic data has TTS bias.** TTS audio is cleaner and acoustically simpler than real speech. If a TTS provider and an STT provider share training data, modeling assumptions, or audio preprocessing pipelines, synthetic audio can make that provider's STT look artificially strong — vendor self-bias. ElevenLabs Scribe evaluated on ElevenLabs-generated audio is a direct example. Synthetic results are treated as reproducibility/control evidence only; **live human voice remains the primary benchmark.** See [docs/methodology.md](docs/methodology.md) for details.
- **No number-word normalization.** Whisper-family models normalize "двадцать" → "20", but our reference text keeps "двадцати". On digit-heavy samples this inflates WER artificially by 5-15 points. We mitigated by reporting "WER clean" without those samples; for "WER full" see `results/consolidated.json`.
- **T-one is out-of-domain on wideband audio.** T-one is a streaming CTC model explicitly trained for telephony (8 kHz narrowband). Our pipeline feeds it 48 kHz `.m4a` files (which its internal `miniaudio` decoder downsamples to 8 kHz, or fails to decode entirely depending on format). It produces poor WER on this benchmark because the clean iPhone audio is completely out of its training distribution. This is a deliberate inclusion to show domain mismatch, not a model bug.
- **RAM measurement is unreliable for mlx-whisper.** mlx uses memory-mapped weights; our peak RSS sampler only sees anonymous pages. Real per-model RAM is likely 1.5-2× higher than reported.
- **NAR models other than Parakeet run on PyTorch CPU without MPS.** GigaAM, Moonshine, Sense Voice — none have an MLX-native port we used here. Their latency numbers reflect "unaccelerated PyTorch on M4 Pro", not "best achievable Apple Silicon latency".
- **Cloud latency includes network round-trip from Hetzner Frankfurt** for ElevenLabs and Groq. Different geos will see different cloud latency.

## Detailed docs

- [docs/methodology.md](docs/methodology.md) — why live voice is the most unbiased source, what synthetic is for, and how TTS-induced bias can distort rankings.
- [docs/commands.md](docs/commands.md) — bench / consolidate / plot invocations.
- [docs/environments.md](docs/environments.md) — two venvs (Python 3.12 vs 3.11) and which models live where.
- [docs/architecture.md](docs/architecture.md) — runner registry, caches, latency metric, how to add a model.

## Files in `results/`

- `consolidated.json` — every (model, sample, run) row deduplicated by latest timestamp. Contains both reference and hypothesis text — you can recompute WER yourself with `wer_eval.py` to verify our numbers.
- `summary.csv` — medians by (model, language). Drop into a spreadsheet.
- `synthetic_consolidated.json` / `synthetic_summary.csv` — latest committed synthetic snapshot used by the GitHub Pages report. Timestamped `results/stt_landscape_2026*.json/csv` files are working outputs and stay ignored.

## License

MIT. Bench code, scripts, and manifest text are MIT-licensed (see `LICENSE`). The findings and methodology summary above are CC-BY 4.0 — credit the repo if you reuse the conclusions.

## Author

Stas Shupilkin — building voice-ai SaaS, runs [@smolevich_voice_bot](https://t.me/smolevich_voice_bot).
- Site: voice.smolevich.com
- LinkedIn: in/stanislav-shupilkin-59482bb4

If you record your own bench using this rig and get different numbers, open an issue or PR — I'd love to compare.
