"""Generate RU synthetic benchmark audio with Silero TTS.

Example:
    TORCH_HOME=.cache/torch .venv-nar/bin/python bench/generate_silero_synthetic.py \
        --voice xenia \
        --sample-rate 48000 \
        --source-manifest samples/manifest.template.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from scipy.io import wavfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=Path("samples/manifest.template.json"))
    parser.add_argument("--voice", default="xenia", choices=["aidar", "baya", "kseniya", "xenia", "eugene"])
    parser.add_argument("--sample-rate", type=int, default=48000, choices=[8000, 24000, 48000])
    parser.add_argument("--model", default="v4_ru")
    parser.add_argument("--out-root", type=Path, default=Path("samples/synthetic"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("samples"))
    return parser.parse_args()


def load_source_samples(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"{path} must contain a samples array")
    return [sample for sample in samples if sample.get("language") == "ru"]


def to_int16(audio: torch.Tensor) -> Any:
    audio = audio.detach().cpu().float().clamp(-1, 1)
    return (audio.numpy() * 32767).astype("int16")


def main() -> int:
    args = parse_args()
    tts_set = f"silero_{args.model}_{args.voice}_{args.sample_rate // 1000}k"
    out_dir = args.out_root / tts_set
    out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)

    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker=args.model,
    )
    model.to("cpu")

    manifest_samples = []
    for sample in load_source_samples(args.source_manifest):
        base_id = str(sample["id"])
        out_path = out_dir / f"{base_id}.wav"
        audio = model.apply_tts(
            text=str(sample["text"]),
            speaker=args.voice,
            sample_rate=args.sample_rate,
            put_accent=True,
            put_yo=True,
        )
        wavfile.write(out_path, args.sample_rate, to_int16(audio))
        manifest_samples.append(
            {
                "id": base_id,
                "language": "ru",
                "path": str(out_path),
                "text": sample["text"],
            }
        )
        print(f"wrote {out_path}")

    manifest = {
        "notes": f"Synthetic RU samples generated with Silero {args.model}, voice={args.voice}, sample_rate={args.sample_rate}.",
        "audio_set_id": tts_set,
        "sample_defaults": {
            "tts_provider": "silero",
            "tts_model": args.model,
            "tts_voice": args.voice,
            "condition": "clean",
            "sample_rate_hz": args.sample_rate,
        },
        "samples": manifest_samples,
    }
    manifest_path = args.manifest_dir / f"manifest_{tts_set}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
