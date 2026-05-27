"""Merge all stt-results/*.json files into one consolidated JSON + summary CSV.

Each bench invocation writes its own timestamped JSON, so RU and EN results
end up scattered across files. This script picks them up, dedups by
(model_id, sample_id, run_index), and emits a single payload + summary.

Dedup rule: when the same triple appears in multiple files, the run from the
file with the latest created_at wins. This way a re-run replaces an earlier
failed/partial attempt without manual cleanup.

Usage:
    python3 bench/consolidate_results.py
    python3 bench/consolidate_results.py --language ru
    python3 bench/consolidate_results.py --include-edge-tts
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


EDGE_TTS_SAMPLES = {"ru-dmitry", "ru-svetlana", "en-aria"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--language", default=None, help="Filter by language (ru, en, …)")
    parser.add_argument(
        "--include-edge-tts",
        action="store_true",
        help="Include synthetic edge-tts samples (ru-dmitry, ru-svetlana, en-aria).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    files = sorted(args.results_dir.glob("stt_landscape_2026*.json"))
    if not files:
        print(f"No result files found in {args.results_dir}")
        return 1

    runs: dict[tuple[str, str, int], dict] = {}
    errors: list[dict] = []
    sample_ids: set[str] = set()
    sources: list[str] = []

    for path in files:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"[skip] {path.name}: bad JSON")
            continue
        sources.append(path.name)
        for r in data.get("runs_detail", []):
            sample_id = r.get("sample_id")
            if not sample_id:
                continue
            if not args.include_edge_tts and sample_id in EDGE_TTS_SAMPLES:
                continue
            if args.language and r.get("language") != args.language:
                continue
            r["_source"] = path.name
            key = (r["model_id"], sample_id, r["run_index"])
            runs[key] = r  # later file (sorted by name=timestamp) wins
            sample_ids.add(sample_id)
        for e in data.get("errors", []):
            errors.append({**e, "source": path.name})

    runs_list = list(runs.values())

    by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in runs_list:
        audio_set_id = str(r.get("audio_set_id") or "live")
        by_key[(r["model_id"], r["language"], audio_set_id)].append(r)
        by_key[(r["model_id"], "all", audio_set_id)].append(r)

    summary = []
    for (model_id, lang, audio_set_id), group in sorted(by_key.items()):
        if not group:
            continue
        first = group[0]
        latencies = [float(r["latency_per_10s"]) for r in group if r.get("latency_per_10s")]
        summary.append({
            "model_id": model_id,
            "runner": first["runner"],
            "arch": first["arch"],
            "language": lang,
            "audio_set_id": audio_set_id,
            "sample_count": len({r["sample_id"] for r in group}),
            "run_count": len(group),
            "median_elapsed_s": round(statistics.median([float(r["elapsed_s"]) for r in group]), 4),
            "median_latency_per_10s": round(statistics.median(latencies), 4) if latencies else None,
            "median_wer_pct": round(statistics.median([float(r["wer_pct"]) for r in group]), 2),
            "median_cer_pct": round(statistics.median([float(r["cer_pct"]) for r in group]), 2),
            "peak_rss_mb": round(max(float(r["peak_rss_mb"]) for r in group), 1),
        })

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = args.out_json or args.results_dir / f"consolidated_{ts}.json"
    out_csv = args.out_csv or args.results_dir / f"consolidated_summary_{ts}.csv"

    payload = {
        "created_at": ts,
        "sources": sources,
        "language_filter": args.language,
        "include_edge_tts": args.include_edge_tts,
        "sample_ids": sorted(sample_ids),
        "runs_detail": runs_list,
        "summary": summary,
        "errors": errors,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    if summary:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            for row in summary:
                writer.writerow(row)

    print(f"Consolidated {len(runs_list)} runs from {len(sources)} files:")
    print(f"  {out_json}")
    print(f"  {out_csv}")
    if errors:
        print(f"  {len(errors)} recorded model errors")
    print()
    print(f'{"model":<30} {"lang":<5} {"set":<28} {"runs":>5} {"lat/10s":>8} {"WER%":>6} {"CER%":>6}')
    print("-" * 100)
    for row in sorted(summary, key=lambda x: (x["audio_set_id"], x["language"] != "all", x["language"], x["median_wer_pct"])):
        lat = f"{row['median_latency_per_10s']:>8.2f}" if row["median_latency_per_10s"] is not None else f"{'—':>8}"
        print(f'{row["model_id"]:<30} {row["language"]:<5} {row["audio_set_id"]:<28} {row["run_count"]:>5} {lat} {row["median_wer_pct"]:>6.1f} {row["median_cer_pct"]:>6.1f}')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
