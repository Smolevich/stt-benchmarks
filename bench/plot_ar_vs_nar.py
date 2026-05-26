"""Generate an AR vs NAR explainer image for the STT landscape article."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/voice-ai-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/voice-ai-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


AR_COLOR = "#2563eb"
NAR_COLOR = "#16a34a"
AUDIO_COLOR = "#94a3b8"
BLANK_COLOR = "#e2e8f0"
TEXT_DARK = "#0f172a"
TEXT_MUTED = "#475569"


def draw_ar(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(
        0.1,
        5.5,
        "AR — autoregressive (Whisper, Canary)",
        fontsize=13,
        fontweight="bold",
        color=AR_COLOR,
    )
    ax.text(
        0.1,
        5.05,
        "каждый токен ждёт предыдущий",
        fontsize=10,
        color=TEXT_MUTED,
        style="italic",
    )

    ax.text(0.1, 4.25, "audio", fontsize=9, color=TEXT_MUTED, va="center")
    ax.add_patch(
        Rectangle(
            (1.2, 4.05),
            8.4,
            0.4,
            facecolor=AUDIO_COLOR,
            edgecolor="none",
        )
    )

    ax.add_patch(
        FancyBboxPatch(
            (3.8, 2.9),
            2.4,
            0.7,
            boxstyle="round,pad=0.05",
            facecolor="white",
            edgecolor=AR_COLOR,
            linewidth=1.5,
        )
    )
    ax.text(5.0, 3.25, "encoder → decoder", fontsize=10, color=AR_COLOR, ha="center", va="center")

    ax.add_patch(
        FancyArrowPatch(
            (5.0, 4.05),
            (5.0, 3.62),
            arrowstyle="-|>",
            mutation_scale=12,
            color=TEXT_MUTED,
            linewidth=1.0,
        )
    )

    ax.text(0.1, 1.6, "output", fontsize=9, color=TEXT_MUTED, va="center")
    tokens = ["t₁", "t₂", "t₃", "t₄", "t₅", "t₆", "t₇"]
    x_start = 1.4
    spacing = 1.15
    for i, tok in enumerate(tokens):
        cx = x_start + i * spacing
        ax.add_patch(
            FancyBboxPatch(
                (cx - 0.32, 1.3),
                0.64,
                0.6,
                boxstyle="round,pad=0.02",
                facecolor=AR_COLOR,
                edgecolor="none",
                alpha=0.85,
            )
        )
        ax.text(cx, 1.6, tok, fontsize=10, color="white", ha="center", va="center", fontweight="bold")

        if i > 0:
            ax.add_patch(
                FancyArrowPatch(
                    (cx - spacing + 0.34, 1.6),
                    (cx - 0.34, 1.6),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    color=AR_COLOR,
                    linewidth=1.2,
                )
            )

    ax.add_patch(
        FancyArrowPatch(
            (5.0, 2.88),
            (5.0, 1.95),
            arrowstyle="-|>",
            mutation_scale=12,
            color=TEXT_MUTED,
            linewidth=1.0,
        )
    )

    ax.text(
        0.1,
        0.55,
        "latency растёт с длиной output  •  GPU обязателен для скорости",
        fontsize=9,
        color=TEXT_MUTED,
        style="italic",
    )


def draw_nar(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(
        0.1,
        5.5,
        "NAR — non-autoregressive (Parakeet, Moonshine, GigaAM)",
        fontsize=13,
        fontweight="bold",
        color=NAR_COLOR,
    )
    ax.text(
        0.1,
        5.05,
        "каждому фрейму аудио — свой output, параллельно",
        fontsize=10,
        color=TEXT_MUTED,
        style="italic",
    )

    ax.text(0.1, 4.25, "audio", fontsize=9, color=TEXT_MUTED, va="center")
    frame_count = 9
    x_start = 1.2
    frame_w = 0.85
    gap = 0.1
    for i in range(frame_count):
        cx = x_start + i * (frame_w + gap)
        ax.add_patch(
            Rectangle(
                (cx, 4.05),
                frame_w,
                0.4,
                facecolor=AUDIO_COLOR,
                edgecolor="white",
                linewidth=1.5,
            )
        )

    ax.text(0.1, 1.6, "output", fontsize=9, color=TEXT_MUTED, va="center")
    outputs = [("t₁", True), ("t₂", True), ("∅", False), ("t₃", True), ("∅", False), ("t₄", True), ("t₅", True), ("∅", False), ("t₆", True)]
    for i, (label, filled) in enumerate(outputs):
        cx = x_start + i * (frame_w + gap)
        ax.add_patch(
            FancyBboxPatch(
                (cx, 1.3),
                frame_w,
                0.6,
                boxstyle="round,pad=0.02",
                facecolor=NAR_COLOR if filled else BLANK_COLOR,
                edgecolor="none",
                alpha=0.85 if filled else 1.0,
            )
        )
        ax.text(
            cx + frame_w / 2,
            1.6,
            label,
            fontsize=10,
            color="white" if filled else TEXT_MUTED,
            ha="center",
            va="center",
            fontweight="bold",
        )

        ax.add_patch(
            FancyArrowPatch(
                (cx + frame_w / 2, 4.0),
                (cx + frame_w / 2, 1.95),
                arrowstyle="-|>",
                mutation_scale=8,
                color=TEXT_MUTED,
                linewidth=0.8,
                alpha=0.6,
            )
        )

    ax.text(
        0.1,
        0.55,
        "latency зависит только от длины input  •  CPU справляется  •  ∅ = blank (тишина)",
        fontsize=9,
        color=TEXT_MUTED,
        style="italic",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/ar_vs_nar.png"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    fig, (ax_ar, ax_nar) = plt.subplots(2, 1, figsize=(10, 7.5))
    fig.patch.set_facecolor("white")

    draw_ar(ax_ar)
    draw_nar(ax_nar)

    plt.subplots_adjust(hspace=0.15, left=0.02, right=0.98, top=0.97, bottom=0.03)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
