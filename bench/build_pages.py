"""Build a static GitHub Pages report from benchmark result JSON files."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DETAILS = "runs_detail"
EXCLUDE_CLEAN_SUFFIXES = ("-04-digits", "-07-names")

# Models that only support a single language — exclude from other-language tables.
# A model producing 100 % WER on a language it doesn't speak adds no signal.
LANGUAGE_ONLY: dict[str, str] = {
    "gigaam-v2-ctc": "ru",
    "gigaam-v2-rnnt": "ru",
    "t-one": "ru",
    "transformers-moonshine-base": "en",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-json", type=Path, default=Path("results/synthetic_consolidated.json"))
    parser.add_argument("--live-json", type=Path, default=Path("results/consolidated.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("site"))
    return parser.parse_args()


def read_runs(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], None
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get(DETAILS, [])), str(data.get("created_at") or "")


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.median(values)), 2)


def base_id(row: dict[str, Any]) -> str:
    value = str(row.get("base_id") or row.get("sample_id") or "")
    return value.split("__", 1)[0]


def is_clean_sample(row: dict[str, Any]) -> bool:
    sample = base_id(row)
    return not any(sample.endswith(suffix) for suffix in EXCLUDE_CLEAN_SUFFIXES)


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        language = str(row.get("language") or "unknown")
        if language == "all":
            continue
        audio_set = str(row.get("audio_set_id") or "live")
        grouped[(audio_set, language, str(row["model_id"]))].append(row)

    rows = []
    for (audio_set, language, model_id), group in grouped.items():
        # Skip language-incompatible models — they produce 100% WER and add no signal.
        supported_lang = LANGUAGE_ONLY.get(model_id)
        if supported_lang is not None and language != supported_lang:
            continue

        clean_group = [row for row in group if is_clean_sample(row)]
        first = group[0]
        latencies = [float(row["latency_per_10s"]) for row in group if row.get("latency_per_10s") is not None]
        peak_rss = [float(row["peak_rss_mb"]) for row in group if row.get("peak_rss_mb") is not None]
        rows.append(
            {
                "audio_set_id": audio_set,
                "language": language,
                "model_id": model_id,
                "runner": first.get("runner", ""),
                "tts_provider": first.get("tts_provider", ""),
                "tts_model": first.get("tts_model", ""),
                "tts_voice": first.get("tts_voice", ""),
                "condition": first.get("condition", ""),
                "sample_count": len({row.get("sample_id") for row in group}),
                "run_count": len(group),
                "median_latency_per_10s": median(latencies),
                "median_wer_pct": median([float(row["wer_pct"]) for row in group]),
                "clean_wer_pct": median([float(row["wer_pct"]) for row in clean_group]),
                "median_cer_pct": median([float(row["cer_pct"]) for row in group]),
                "peak_rss_mb": max(peak_rss) if peak_rss else None,
            }
        )
    return sorted(rows, key=lambda row: (row["audio_set_id"], row["language"], row["clean_wer_pct"] or 999, row["median_wer_pct"] or 999))


def esc(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def fmt(value: object, suffix: str = "") -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def table(rows: list[dict[str, Any]], title: str) -> str:
    if not rows:
        return f"<section><h2>{esc(title)}</h2><p>No data yet.</p></section>"
    body = "\n".join(
        "<tr>"
        f"<td>{esc(row['audio_set_id'])}</td>"
        f"<td>{esc(row['language'])}</td>"
        f"<td>{esc(row['model_id'])}</td>"
        f"<td class='col-acc'>{esc(row.get('tts_provider') or '—')}</td>"
        f"<td class='col-acc'>{esc(row.get('tts_model') or '—')}</td>"
        f"<td class='col-acc'>{esc(row.get('tts_voice') or '—')}</td>"
        f"<td class='col-acc'>{fmt(row.get('clean_wer_pct'), '%')}</td>"
        f"<td class='col-acc'>{fmt(row.get('median_wer_pct'), '%')}</td>"
        f"<td class='col-acc'>{fmt(row.get('median_cer_pct'), '%')}</td>"
        f"<td class='col-res'>{fmt(row.get('median_latency_per_10s'), ' s')}</td>"
        f"<td class='col-res'>{fmt(row.get('peak_rss_mb'))}</td>"
        f"<td>{esc(row['run_count'])}</td>"
        "</tr>"
        for row in rows
    )
    return f"""
<section>
  <h2>{esc(title)}</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Audio set</th>
          <th>Lang</th>
          <th>STT model</th>
          <th class="col-acc">TTS provider</th>
          <th class="col-acc">TTS model</th>
          <th class="col-acc">Voice</th>
          <th class="col-acc">WER clean</th>
          <th class="col-acc">WER full</th>
          <th class="col-acc">CER</th>
          <th class="col-res">Latency / 10s</th>
          <th class="col-res">Peak RAM (MB)</th>
          <th>Runs</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
  </div>
</section>
"""


def language_options(synthetic_rows: list[dict[str, Any]], live_rows: list[dict[str, Any]]) -> str:
    languages = sorted({str(row["language"]) for row in [*live_rows, *synthetic_rows] if row.get("language")})
    options = ['<option value="all">All languages</option>']
    options.extend(f'<option value="{esc(language)}">{esc(language.upper())}</option>' for language in languages)
    return "\n".join(options)


def audio_set_options(synthetic_rows: list[dict[str, Any]], live_rows: list[dict[str, Any]]) -> str:
    audio_sets = sorted({str(row["audio_set_id"]) for row in [*live_rows, *synthetic_rows] if row.get("audio_set_id")})
    options = ['<option value="all">All audio sets</option>']
    options.extend(f'<option value="{esc(aset)}">{esc(aset)}</option>' for aset in audio_sets)
    return "\n".join(options)


def build_html(synthetic_rows: list[dict[str, Any]], live_rows: list[dict[str, Any]], synthetic_created: str | None, live_created: str | None) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lang_options = language_options(synthetic_rows, live_rows)
    aset_options = audio_set_options(synthetic_rows, live_rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STT Benchmark Results</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --text: #17202a;
      --muted: #5b6673;
      --line: #d9dee7;
      --head: #eef2f6;
      --accent: #0b6bcb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    header {{
      margin-bottom: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 12px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    p {{
      max-width: 860px;
      margin: 6px 0;
      color: var(--muted);
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 14px;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 22px 0 8px;
      color: var(--muted);
      font-size: 14px;
    }}
    select {{
      height: 34px;
      padding: 0 34px 0 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--text);
      font: inherit;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      background: white;
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      min-width: 980px;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 9px 10px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }}
    th {{
      background: var(--head);
      font-weight: 650;
    }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    a {{ color: var(--accent); }}
    body[data-view="accuracy"] .col-res {{ display: none; }}
    body[data-view="resources"] .col-acc {{ display: none; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>STT Benchmark Results</h1>
      <p>Median WER/CER and latency grouped by audio set, language, and STT model. WER clean excludes the digits and proper-name stress samples. Audio files are not published. <strong>Runs</strong> = (number of unique samples) × (repetitions).</p>
      <div class="meta">
        <span>Generated: {esc(generated_at)}</span>
        <span>Synthetic data: {esc(synthetic_created or "not available")}</span>
        <span>Live data: {esc(live_created or "not available")}</span>
      </div>
      <div class="controls">
        <label>
          <span>View</span>
          <select id="view-filter">
            <option value="accuracy">Accuracy (WER/CER)</option>
            <option value="resources">Resources (RAM/Latency)</option>
            <option value="all">All columns</option>
          </select>
        </label>
        <label>
          <span>Audio Set</span>
          <select id="audio-set-filter">
            {aset_options}
          </select>
        </label>
        <label>
          <span>Language</span>
          <select id="language-filter">
            {lang_options}
          </select>
        </label>
      </div>
    </header>
    {table(live_rows, "Live Voice")}
    {table(synthetic_rows, "Synthetic Audio")}
  </main>
  <script>
    const langFilter = document.getElementById("language-filter");
    const audioSetFilter = document.getElementById("audio-set-filter");
    const rows = Array.from(document.querySelectorAll("tbody tr"));

    function applyFilters() {{
      const selectedLang = langFilter.value;
      const selectedAudioSet = audioSetFilter.value;
      
      for (const row of rows) {{
        const audioSet = row.cells[0]?.textContent.trim();
        const language = row.cells[1]?.textContent.trim();
        
        const matchLang = selectedLang === "all" || language === selectedLang;
        const matchAudioSet = selectedAudioSet === "all" || audioSet === selectedAudioSet;
        
        row.hidden = !(matchLang && matchAudioSet);
      }}
    }}

    langFilter.addEventListener("change", applyFilters);
    audioSetFilter.addEventListener("change", applyFilters);
    applyFilters();

    const viewFilter = document.getElementById("view-filter");
    function applyView() {{
      document.body.setAttribute("data-view", viewFilter.value);
    }}
    viewFilter.addEventListener("change", applyView);
    applyView();
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    synthetic_runs, synthetic_created = read_runs(args.synthetic_json)
    live_runs, live_created = read_runs(args.live_json)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.synthetic_json.exists():
        shutil.copy2(args.synthetic_json, data_dir / args.synthetic_json.name)
    if args.live_json.exists():
        shutil.copy2(args.live_json, data_dir / args.live_json.name)

    synthetic_rows = summarize(synthetic_runs)
    live_rows = summarize(live_runs)
    (data_dir / "synthetic_summary.json").write_text(json.dumps(synthetic_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (data_dir / "live_summary.json").write_text(json.dumps(live_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "index.html").write_text(
        build_html(synthetic_rows, live_rows, synthetic_created, live_created),
        encoding="utf-8",
    )
    print(f"Wrote {args.out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
