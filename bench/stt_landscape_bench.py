"""Benchmark local and cloud STT models for the STT landscape article.

The script intentionally talks to model libraries directly. Handy is useful as
a GUI reference, but it does not expose a stable CLI/API for repeatable evals.

Examples:
    python3 bench/stt_landscape_bench.py --list-models

    python3 bench/stt_landscape_bench.py \
        --samples samples/manifest.example.json \
        --models mlx-whisper-small \
        --warmup 1 --runs 3

    GROQ_API_KEY=... python3 bench/stt_landscape_bench.py \
        --samples samples/manifest.json \
        --models mlx-whisper-small,mlx-whisper-large-v3-turbo,groq-whisper-large-v3-turbo
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import mimetypes
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil

sys.path.append(str(Path(__file__).resolve().parent))
from wer_eval import compute_cer, compute_wer  # noqa: E402


DEFAULT_SAMPLE_MANIFEST = Path("samples/manifest.json")
FALLBACK_SAMPLE_MANIFEST = Path("samples/manifest.example.json")
DEFAULT_OUT_DIR = Path("results")
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"
SAMPLE_METADATA_FIELDS = [
    "base_id",
    "audio_set_id",
    "tts_provider",
    "tts_model",
    "tts_voice",
    "condition",
    "sample_rate_hz",
    "ambient",
]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    runner: str
    model: str
    arch: str
    languages: str
    disk_hint: str = ""
    requires_env: str = ""
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "mlx-whisper-small": ModelSpec(
        id="mlx-whisper-small",
        runner="mlx_whisper",
        model="mlx-community/whisper-small-mlx",
        arch="AR",
        languages="99",
        disk_hint="~244 MB",
    ),
    "mlx-whisper-medium": ModelSpec(
        id="mlx-whisper-medium",
        runner="mlx_whisper",
        model="mlx-community/whisper-medium-mlx",
        arch="AR",
        languages="99",
        disk_hint="~1.5 GB",
    ),
    "mlx-whisper-large-v3-turbo": ModelSpec(
        id="mlx-whisper-large-v3-turbo",
        runner="mlx_whisper",
        model="mlx-community/whisper-large-v3-turbo",
        arch="AR",
        languages="99",
        disk_hint="~1.6 GB",
    ),
    "mlx-whisper-large-v3": ModelSpec(
        id="mlx-whisper-large-v3",
        runner="mlx_whisper",
        model="mlx-community/whisper-large-v3-mlx",
        arch="AR",
        languages="99",
        disk_hint="~3.1 GB",
    ),
    "groq-whisper-large-v3-turbo": ModelSpec(
        id="groq-whisper-large-v3-turbo",
        runner="groq",
        model="whisper-large-v3-turbo",
        arch="AR",
        languages="99",
        requires_env="GROQ_API_KEY",
        notes="Cloud API baseline used by the bot.",
    ),
    "elevenlabs-scribe-v1": ModelSpec(
        id="elevenlabs-scribe-v1",
        runner="elevenlabs",
        model="scribe_v1",
        arch="AR",
        languages="99",
        requires_env="ELEVENLABS_API_KEY",
        notes="ElevenLabs Scribe v1 STT API.",
    ),
    "elevenlabs-scribe-v1-experimental": ModelSpec(
        id="elevenlabs-scribe-v1-experimental",
        runner="elevenlabs",
        model="scribe_v1_experimental",
        arch="AR",
        languages="99",
        requires_env="ELEVENLABS_API_KEY",
        notes="ElevenLabs Scribe v1 experimental — newer, stronger on RU.",
    ),
    "nemo-parakeet-tdt-0.6b-v2": ModelSpec(
        id="nemo-parakeet-tdt-0.6b-v2",
        runner="nemo",
        model="nvidia/parakeet-tdt-0.6b-v2",
        arch="NAR (TDT)",
        languages="EN",
        disk_hint="~600 MB",
        notes="Requires nemo_toolkit[asr].",
    ),
    "nemo-parakeet-tdt-0.6b-v3": ModelSpec(
        id="nemo-parakeet-tdt-0.6b-v3",
        runner="nemo",
        model="nvidia/parakeet-tdt-0.6b-v3",
        arch="NAR (TDT)",
        languages="25",
        disk_hint="~600 MB",
        notes="Requires nemo_toolkit[asr].",
    ),
    "parakeet-tdt-0.6b-v3": ModelSpec(
        id="parakeet-tdt-0.6b-v3",
        runner="transformers",
        model="nvidia/parakeet-tdt-0.6b-v3",
        arch="NAR (TDT)",
        languages="25",
        disk_hint="~2.4 GB",
        notes="Via transformers + librosa (no NeMo needed).",
    ),
    "parakeet-tdt-0.6b-v3-mlx": ModelSpec(
        id="parakeet-tdt-0.6b-v3-mlx",
        runner="parakeet_mlx",
        model="mlx-community/parakeet-tdt-0.6b-v3",
        arch="NAR (TDT)",
        languages="25",
        disk_hint="~600 MB",
        notes="Apple Silicon native via parakeet-mlx (community port).",
    ),
    "nemo-canary-180m-flash": ModelSpec(
        id="nemo-canary-180m-flash",
        runner="nemo",
        model="nvidia/canary-180m-flash",
        arch="AR",
        languages="4 EU",
        disk_hint="~360 MB",
        notes="Requires nemo_toolkit[asr].",
    ),
    "nemo-canary-1b-v2": ModelSpec(
        id="nemo-canary-1b-v2",
        runner="nemo",
        model="nvidia/canary-1b-v2",
        arch="AR",
        languages="4 EU",
        disk_hint="~2 GB",
        notes="Requires nemo_toolkit[asr].",
    ),
    "transformers-moonshine-base": ModelSpec(
        id="transformers-moonshine-base",
        runner="transformers",
        model="UsefulSensors/moonshine-base",
        arch="NAR",
        languages="EN",
        disk_hint="~150 MB",
        notes="Requires transformers + torch.",
    ),
    "gigaam-v2-ctc": ModelSpec(
        id="gigaam-v2-ctc",
        runner="gigaam",
        model="v2_ctc",
        arch="NAR (CTC)",
        languages="RU",
        notes="GigaAM v2 CTC from Sber (v3 not yet in pip package).",
    ),
    "gigaam-v2-rnnt": ModelSpec(
        id="gigaam-v2-rnnt",
        runner="gigaam",
        model="v2_rnnt",
        arch="NAR (RNN-T)",
        languages="RU",
        notes="GigaAM v2 RNN-T from Sber.",
    ),
    "sensevoice-small": ModelSpec(
        id="sensevoice-small",
        runner="funasr",
        model="iic/SenseVoiceSmall",
        arch="NAR (CTC)",
        languages="50+ (works on EN, breaks on RU via funasr API)",
        disk_hint="~900 MB",
        notes="Alibaba via funasr. Auto-detect language, ignores language hint.",
    ),
    "t-one": ModelSpec(
        id="t-one",
        runner="tone",
        model="t-tech/T-one",
        arch="NAR (CTC)",
        languages="RU (telephony 8kHz)",
        disk_hint="~290 MB",
        notes="T-Bank / T-Tech streaming Conformer-CTC. Install: pip install git+https://github.com/voicekit-team/T-one.git",
    ),
}


class PeakMemorySampler:
    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self.process = psutil.Process(os.getpid())
        self.peak_bytes = 0
        self.running = False
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def __enter__(self) -> "PeakMemorySampler":
        self.running = True
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.running = False
        self.thread.join(timeout=1)

    def sample(self) -> None:
        while self.running:
            try:
                rss = self.process.memory_info().rss
                for child in self.process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.Error:
                        continue
                self.peak_bytes = max(self.peak_bytes, rss)
            except psutil.Error:
                pass
            time.sleep(self.interval_s)


class RunnerCache:
    def __init__(self) -> None:
        self.loaded: dict[str, Callable[[str, str], str]] = {}

    def get(self, spec: ModelSpec) -> Callable[[str, str], str]:
        if spec.id not in self.loaded:
            self.loaded[spec.id] = build_runner(spec)
        return self.loaded[spec.id]


def build_runner(spec: ModelSpec) -> Callable[[str, str], str]:
    if spec.requires_env and not os.environ.get(spec.requires_env):
        raise RuntimeError(f"{spec.id} requires ${spec.requires_env}")

    if spec.runner == "mlx_whisper":
        import mlx_whisper

        def transcribe(path: str, language: str) -> str:
            result = mlx_whisper.transcribe(
                path,
                path_or_hf_repo=spec.model,
                language=None if language == "auto" else language,
            )
            return str(result.get("text", "")).strip()

        return transcribe

    if spec.runner == "groq":
        import httpx

        def transcribe(path: str, language: str) -> str:
            with open(path, "rb") as audio_file:
                media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
                files = {"file": (Path(path).name, audio_file, media_type)}
                data: dict[str, str] = {
                    "model": spec.model,
                    "response_format": "json",
                    "temperature": "0",
                }
                if language != "auto":
                    data["language"] = language
                response = httpx.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
                    files=files,
                    data=data,
                    timeout=600,
                )
            response.raise_for_status()
            return str(response.json().get("text", "")).strip()

        return transcribe

    if spec.runner == "elevenlabs":
        import httpx

        def transcribe(path: str, language: str) -> str:
            with open(path, "rb") as audio_file:
                media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
                files = {"file": (Path(path).name, audio_file, media_type)}
                data: dict[str, str] = {
                    "model_id": spec.model,
                }
                if language != "auto":
                    data["language_code"] = language
                response = httpx.post(
                    ELEVENLABS_URL,
                    headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                    files=files,
                    data=data,
                    timeout=600,
                )
            response.raise_for_status()
            return str(response.json().get("text", "")).strip()

        return transcribe

    if spec.runner == "nemo":
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise RuntimeError("Install NeMo first: pip install 'nemo_toolkit[asr]'") from exc

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=spec.model)
        if hasattr(model, "eval"):
            model.eval()

        def transcribe(path: str, language: str) -> str:
            result = model.transcribe([path], batch_size=1)
            item = result[0] if isinstance(result, list) else result
            if hasattr(item, "text"):
                return str(item.text).strip()
            return str(item).strip()

        return transcribe

    if spec.runner == "transformers":
        try:
            from transformers import pipeline
            import librosa
        except ImportError as exc:
            raise RuntimeError("Install transformers + librosa first: pip install transformers librosa") from exc

        asr = pipeline("automatic-speech-recognition", model=spec.model)
        target_sr = asr.feature_extractor.sampling_rate

        def transcribe(path: str, language: str) -> str:
            audio, _ = librosa.load(path, sr=target_sr, mono=True)
            result = asr({"array": audio, "sampling_rate": target_sr})
            return str(result.get("text", "")).strip()

        return transcribe

    if spec.runner == "gigaam":
        try:
            import gigaam
        except ImportError as exc:
            raise RuntimeError("Install GigaAM first: pip install gigaam") from exc

        model = gigaam.load_model(spec.model)

        def transcribe(path: str, language: str) -> str:
            return str(model.transcribe(path)).strip()

        return transcribe

    if spec.runner == "parakeet_mlx":
        try:
            from parakeet_mlx import from_pretrained
        except ImportError as exc:
            raise RuntimeError("Install parakeet-mlx first: pip install parakeet-mlx") from exc

        model = from_pretrained(spec.model)

        def transcribe(path: str, language: str) -> str:
            result = model.transcribe(path)
            text = result.text if hasattr(result, "text") else str(result)
            return str(text).strip()

        return transcribe

    if spec.runner == "funasr":
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError("Install funasr first: pip install funasr") from exc
        import re

        model = AutoModel(model=spec.model, disable_update=True, trust_remote_code=True)
        # Sense Voice prepends tags like <|en|><|NEUTRAL|><|Speech|><|withitn|> — strip them.
        tag_re = re.compile(r"<\|[^|]*\|>")

        def transcribe(path: str, language: str) -> str:
            result = model.generate(input=path, use_itn=True)
            if not result:
                return ""
            text = str(result[0].get("text", "") if isinstance(result, list) else result).strip()
            return tag_re.sub("", text).strip()

        return transcribe

    if spec.runner == "tone":
        try:
            from tone import StreamingCTCPipeline, read_audio
        except ImportError as exc:
            raise RuntimeError(
                "Install T-one first: pip install git+https://github.com/voicekit-team/T-one.git"
            ) from exc

        pipeline = StreamingCTCPipeline.from_hugging_face()

        def transcribe(path: str, language: str) -> str:
            # Fallback to .wav if the original is an unsupported format (like .m4a)
            if path.endswith(".m4a") or path.endswith(".mp3"):
                wav_path = path.rsplit(".", 1)[0] + ".wav"
                import os
                if os.path.exists(wav_path):
                    path = wav_path
            audio = read_audio(path)
            result = pipeline.forward_offline(audio)
            if isinstance(result, list):
                return " ".join(getattr(seg, "text", str(seg)) for seg in result).strip()
            return str(result).strip()

        return transcribe

    raise RuntimeError(f"Unknown runner: {spec.runner}")


def normalize_path(path: str, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def load_samples(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path} must contain a non-empty 'samples' array")

    prepared = []
    base_dir = Path.cwd()
    manifest_audio_set_id = manifest.get("audio_set_id")
    sample_defaults = manifest.get("sample_defaults", {})
    if sample_defaults is None:
        sample_defaults = {}
    if not isinstance(sample_defaults, dict):
        raise ValueError(f"{path} sample_defaults must be an object when present")

    for sample in samples:
        sample_path = normalize_path(str(sample["path"]), base_dir)
        if not sample_path.exists():
            raise FileNotFoundError(sample_path)
        base_id = str(sample.get("base_id") or sample["id"])
        audio_set_id = sample.get("audio_set_id") or manifest_audio_set_id
        sample_id = str(sample["id"])
        if audio_set_id and "__" not in sample_id:
            sample_id = f"{base_id}__{audio_set_id}"

        metadata = {
            key: sample.get(key, sample_defaults.get(key))
            for key in SAMPLE_METADATA_FIELDS
        }
        metadata["base_id"] = base_id
        metadata["audio_set_id"] = audio_set_id
        if "tts_model" not in sample and "_tts_model" in sample:
            metadata["tts_model"] = sample["_tts_model"]
        if "ambient" not in sample and "_ambient" in sample:
            metadata["ambient"] = sample["_ambient"]

        item = {
            "id": sample_id,
            "language": str(sample.get("language", "auto")),
            "path": str(sample_path),
            "text": str(sample["text"]),
            "duration_s": audio_duration_s(sample_path),
        }
        item.update({key: value for key, value in metadata.items() if value is not None})
        prepared.append(item)
    return prepared


def audio_duration_s(path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return round(float(result.stdout.strip()), 3)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024 * 1024), 2)


def directory_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def model_cache_size_bytes(spec: ModelSpec) -> int | None:
    repo_path = spec.model.replace("/", "--")
    candidates = [
        Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_path}",
        Path.home() / ".cache" / "mlx_whisper" / spec.model,
    ]
    sizes = [directory_size_bytes(path) for path in candidates if path.exists()]
    return max(sizes) if sizes else None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.median(values)), 4)


def pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100, 3)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def benchmark_model(
    spec: ModelSpec,
    samples: list[dict[str, Any]],
    runner_cache: RunnerCache,
    warmup: int,
    runs: int,
    cloud_sleep_s: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transcribe = runner_cache.get(spec)
    run_rows: list[dict[str, Any]] = []
    is_cloud = spec.runner in {"groq", "elevenlabs"}
    pause = cloud_sleep_s if is_cloud else 0.0

    for sample in samples:
        for warmup_index in range(warmup):
            if pause:
                time.sleep(pause)
            print(f"[warmup] {spec.id} {sample['id']} #{warmup_index + 1}", flush=True)
            transcribe(sample["path"], sample["language"])

        for run_index in range(runs):
            if pause:
                time.sleep(pause)
            print(f"[run] {spec.id} {sample['id']} #{run_index + 1}", flush=True)
            gc.collect()
            with PeakMemorySampler() as memory:
                started = time.monotonic()
                hypothesis = transcribe(sample["path"], sample["language"])
                elapsed_s = time.monotonic() - started

            duration_s = sample["duration_s"]
            latency_per_10s = None
            if duration_s and duration_s > 0:
                latency_per_10s = elapsed_s / duration_s * 10

            run_rows.append(
                {
                    "model_id": spec.id,
                    "runner": spec.runner,
                    "arch": spec.arch,
                    "model": spec.model,
                    "sample_id": sample["id"],
                    "base_id": sample.get("base_id", sample["id"]),
                    "audio_set_id": sample.get("audio_set_id"),
                    "tts_provider": sample.get("tts_provider"),
                    "tts_model": sample.get("tts_model"),
                    "tts_voice": sample.get("tts_voice"),
                    "condition": sample.get("condition"),
                    "sample_rate_hz": sample.get("sample_rate_hz"),
                    "ambient": sample.get("ambient"),
                    "language": sample["language"],
                    "duration_s": duration_s,
                    "run_index": run_index + 1,
                    "elapsed_s": round(elapsed_s, 4),
                    "latency_per_10s": round(latency_per_10s, 4) if latency_per_10s else None,
                    "wer_pct": pct(compute_wer(sample["text"], hypothesis)),
                    "cer_pct": pct(compute_cer(sample["text"], hypothesis)),
                    "peak_rss_mb": bytes_to_mb(memory.peak_bytes),
                    "reference": sample["text"],
                    "hypothesis": hypothesis,
                }
            )

    return run_rows, summarize_model(spec, run_rows)


def summarize_model(spec: ModelSpec, run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    languages = sorted({str(row["language"]) for row in run_rows})
    disk_bytes = model_cache_size_bytes(spec)
    audio_sets = sorted({str(row.get("audio_set_id") or "") for row in run_rows if row.get("audio_set_id")})
    audio_set_id = audio_sets[0] if len(audio_sets) == 1 else ("mixed" if audio_sets else None)

    for language in languages:
        lang_rows = [row for row in run_rows if row["language"] == language]
        rows.append(
            {
                "model_id": spec.id,
                "runner": spec.runner,
                "arch": spec.arch,
                "model": spec.model,
                "language": language,
                "audio_set_id": audio_set_id,
                "sample_count": len({row["sample_id"] for row in lang_rows}),
                "run_count": len(lang_rows),
                "median_elapsed_s": median([float(row["elapsed_s"]) for row in lang_rows]),
                "median_latency_per_10s": median(
                    [float(row["latency_per_10s"]) for row in lang_rows if row["latency_per_10s"] is not None]
                ),
                "median_wer_pct": median([float(row["wer_pct"]) for row in lang_rows]),
                "median_cer_pct": median([float(row["cer_pct"]) for row in lang_rows]),
                "peak_rss_mb": max(float(row["peak_rss_mb"]) for row in lang_rows),
                "disk_mb": bytes_to_mb(disk_bytes),
                "disk_hint": spec.disk_hint,
                "notes": spec.notes,
            }
        )

    rows.append(
        {
            "model_id": spec.id,
            "runner": spec.runner,
            "arch": spec.arch,
            "model": spec.model,
            "language": "all",
            "audio_set_id": audio_set_id,
            "sample_count": len({row["sample_id"] for row in run_rows}),
            "run_count": len(run_rows),
            "median_elapsed_s": median([float(row["elapsed_s"]) for row in run_rows]),
            "median_latency_per_10s": median(
                [float(row["latency_per_10s"]) for row in run_rows if row["latency_per_10s"] is not None]
            ),
            "median_wer_pct": median([float(row["wer_pct"]) for row in run_rows]),
            "median_cer_pct": median([float(row["cer_pct"]) for row in run_rows]),
            "peak_rss_mb": max(float(row["peak_rss_mb"]) for row in run_rows),
            "disk_mb": bytes_to_mb(disk_bytes),
            "disk_hint": spec.disk_hint,
            "notes": spec.notes,
        }
    )
    return rows


def parse_models(value: str) -> list[ModelSpec]:
    if value == "all":
        return list(MODEL_REGISTRY.values())
    specs = []
    for model_id in value.split(","):
        key = model_id.strip()
        if key not in MODEL_REGISTRY:
            known = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown model '{key}'. Known models: {known}")
        specs.append(MODEL_REGISTRY[key])
    return specs


def resolve_sample_manifest(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_SAMPLE_MANIFEST and FALLBACK_SAMPLE_MANIFEST.exists():
        print(
            f"Warning: {DEFAULT_SAMPLE_MANIFEST} not found, using {FALLBACK_SAMPLE_MANIFEST} smoke-test samples.",
            file=sys.stderr,
        )
        return FALLBACK_SAMPLE_MANIFEST
    raise FileNotFoundError(path)


def print_models() -> None:
    print(f"{'id':34} {'runner':14} {'arch':10} {'lang':8} model")
    print("-" * 96)
    for spec in MODEL_REGISTRY.values():
        print(f"{spec.id:34} {spec.runner:14} {spec.arch:10} {spec.languages:8} {spec.model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--models", default="mlx-whisper-small")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--cloud-sleep-s",
        type=float,
        default=0.0,
        help="Sleep N seconds before each cloud (groq/elevenlabs) request to dodge rate limits.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_models:
        print_models()
        return 0

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    sample_manifest = resolve_sample_manifest(args.samples)
    samples = load_samples(sample_manifest)
    models = parse_models(args.models)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    runner_cache = RunnerCache()

    for spec in models:
        try:
            model_run_rows, model_summary_rows = benchmark_model(
                spec=spec,
                samples=samples,
                runner_cache=runner_cache,
                warmup=args.warmup,
                runs=args.runs,
                cloud_sleep_s=args.cloud_sleep_s,
            )
            run_rows.extend(model_run_rows)
            summary_rows.extend(model_summary_rows)
        except Exception as exc:
            errors.append({"model_id": spec.id, "error": str(exc)})
            print(f"[error] {spec.id}: {exc}", file=sys.stderr, flush=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "created_at": timestamp,
        "sample_manifest": str(sample_manifest),
        "warmup": args.warmup,
        "runs": args.runs,
        "models": [spec.__dict__ for spec in models],
        "summary": summary_rows,
        "runs_detail": run_rows,
        "errors": errors,
    }

    json_path = args.out_dir / f"stt_landscape_{timestamp}.json"
    runs_csv_path = args.out_dir / f"stt_landscape_runs_{timestamp}.csv"
    summary_csv_path = args.out_dir / f"stt_landscape_summary_{timestamp}.csv"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(
        runs_csv_path,
        run_rows,
        [
            "model_id",
            "runner",
            "arch",
            "model",
            "sample_id",
            "base_id",
            "audio_set_id",
            "tts_provider",
            "tts_model",
            "tts_voice",
            "condition",
            "sample_rate_hz",
            "ambient",
            "language",
            "duration_s",
            "run_index",
            "elapsed_s",
            "latency_per_10s",
            "wer_pct",
            "cer_pct",
            "peak_rss_mb",
            "reference",
            "hypothesis",
        ],
    )
    write_csv(
        summary_csv_path,
        summary_rows,
        [
            "model_id",
            "runner",
            "arch",
            "model",
            "language",
            "audio_set_id",
            "sample_count",
            "run_count",
            "median_elapsed_s",
            "median_latency_per_10s",
            "median_wer_pct",
            "median_cer_pct",
            "peak_rss_mb",
            "disk_mb",
            "disk_hint",
            "notes",
        ],
    )

    print(f"Wrote {json_path}")
    print(f"Wrote {runs_csv_path}")
    print(f"Wrote {summary_csv_path}")
    if errors:
        print(f"Completed with {len(errors)} model error(s). See JSON for details.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
