"""Generate a latency x WER scatter plot from stt_landscape_summary CSV."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/voice-ai-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/voice-ai-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ARCH_COLORS = {
    "AR": "#2563eb",
    "NAR": "#16a34a",
    "NAR (TDT)": "#16a34a",
    "NAR (CTC)": "#15803d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--language", default="all")
    parser.add_argument("--out", type=Path, default=Path("experiments/stt-results/stt_latency_wer.png"))
    return parser.parse_args()


def load_rows(path: Path, language: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [
        row
        for row in rows
        if row.get("language") == language
        and row.get("median_latency_per_10s")
        and row.get("median_wer_pct")
    ]


def arch_color(arch: str) -> str:
    return ARCH_COLORS.get(arch, "#64748b")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.summary_csv, args.language)
    if not rows:
        raise SystemExit(f"No plottable rows for language={args.language}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

    for row in rows:
        x = float(row["median_latency_per_10s"])
        y = float(row["median_wer_pct"])
        arch = row.get("arch", "")
        ax.scatter(x, y, s=80, color=arch_color(arch), alpha=0.9, edgecolor="white", linewidth=0.8)
        ax.annotate(
            row["model_id"].replace("mlx-", "").replace("nemo-", ""),
            (x, y),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title(f"STT latency vs WER ({args.language})")
    ax.set_xlabel("Median latency, seconds per 10s audio")
    ax.set_ylabel("Median WER, %")
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = []
    labels = []
    for arch in sorted({row.get("arch", "") for row in rows}):
        handle = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=arch,
            markerfacecolor=arch_color(arch),
            markersize=8,
        )
        handles.append(handle)
        labels.append(arch)
    ax.legend(handles, labels, title="Architecture", loc="best")

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
