# Path mismatch caveat

`bench/stt_landscape_bench.py` carries leftovers from the upstream voice-ai repo this was extracted from:

- `from experiments.wer_eval import compute_cer, compute_wer` (line 40) — there is no `experiments/` directory in this repo, so this import will fail. `wer_eval.py` lives in `bench/`. To run the bench, either change the import to `from wer_eval import ...` (and adjust the `sys.path.append` accordingly) or add an `experiments/` shim that re-exports the symbols.
- `DEFAULT_SAMPLE_MANIFEST = Path("experiments/stt_samples.json")` and `DEFAULT_OUT_DIR = Path("experiments/stt-results")` — always pass `--samples samples/manifest.json` and `--out-dir results/` explicitly (the README commands already do this).
- The docstring examples at the top of the file also reference `experiments/...` — ignore them, follow the commands in [README.md](../README.md) or [docs/commands.md](commands.md).

The README's reproduction steps invoke the bench correctly but don't mention the broken import, so a fresh clone will fail at import time until that line is patched.
