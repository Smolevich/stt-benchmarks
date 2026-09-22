"""Шумные варианты синтетических сэмплов из чистых TTS-мастеров.

Шум — шаг бенча, а не свойство файла: в репозитории лежит чистый мастер, уровень
и характер шума описаны рецептом здесь. Поэтому уровень можно поменять или
добавить новый, не переозвучивая текст (и не тратя символы TTS).

Генератор шума получает фиксированный seed: без него каждый прогон давал бы
другое аудио и WER на одном и том же кейсе гулял бы от запуска к запуску.

    python3 bench/mix_noise.py            # перегенерировать все кейсы
    python3 bench/mix_noise.py --list     # показать рецепты

Мастера и результат лежат в samples/synthetic/<audio_set_id>/, манифест —
samples/manifest_synthetic_noise.json.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_SET_ID = "elevenlabs_sarah_noise"
SET_DIR = ROOT / "samples" / "synthetic" / AUDIO_SET_ID
SAMPLE_RATE_HZ = 44100
MP3_BITRATE = "128k"
# Потолок пика после суммирования дорожек: mp3 без запаса клиппится, и тогда
# измеряешь искажения кодека, а не шум.
PEAK_CEILING_DB = -3.0
TONE_FREQ_HZ = 3500


@dataclass(frozen=True)
class NoiseCase:
    """Один шумный кейс: мастер плюс рецепт шума."""

    out_id: str
    master: str
    # Насколько речь громче шума по среднему уровню. Отрицательное — шум громче речи.
    snr_db: float
    # Узкополосный дребезжащий тон; None — только белый шум.
    tone_below_noise_db: float | None
    lead_pad_s: float
    tail_pad_s: float
    seed: int
    noise_level: str


CASES: tuple[NoiseCase, ...] = (
    NoiseCase(
        out_id="ru-02-noise",
        master="ru-01-master.mp3",
        snr_db=8.0,
        tone_below_noise_db=10.0,
        lead_pad_s=1.5,
        tail_pad_s=2.0,
        seed=20260922,
        noise_level="moderate",
    ),
    NoiseCase(
        out_id="hy-02-noise",
        master="hy-01-master.mp3",
        snr_db=8.0,
        tone_below_noise_db=None,
        lead_pad_s=2.5,
        tail_pad_s=3.0,
        seed=20260923,
        noise_level="moderate",
    ),
    NoiseCase(
        out_id="hy-03-heavy",
        master="hy-01-master.mp3",
        snr_db=-3.0,
        tone_below_noise_db=10.0,
        lead_pad_s=2.5,
        tail_pad_s=3.0,
        seed=20260924,
        noise_level="heavy",
    ),
)


def ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", *args],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed: {' '.join(args)}\n{result.stderr[-2000:]}")
    return result


def mean_volume_db(path: Path) -> float:
    """Средний уровень дорожки по volumedetect — от него считаются уровни шума."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect",
                             "-f", "null", "-"], capture_output=True, text=True)
    match = re.search(r"mean_volume:\s*(-?[\d.]+) dB", result.stderr)
    if not match:
        raise RuntimeError(f"volumedetect не дал mean_volume для {path}")
    return float(match.group(1))


def peak_volume_db(path: Path) -> float:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect",
                             "-f", "null", "-"], capture_output=True, text=True)
    match = re.search(r"max_volume:\s*(-?[\d.]+) dB", result.stderr)
    if not match:
        raise RuntimeError(f"volumedetect не дал max_volume для {path}")
    return float(match.group(1))


def duration_s(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                            capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def write_white_noise(seconds: float, seed: int, out: Path) -> None:
    ffmpeg(["-f", "lavfi",
            "-i", f"anoisesrc=color=white:sample_rate={SAMPLE_RATE_HZ}:duration={seconds}"
                  f":amplitude=1:seed={seed}",
            "-ac", "1", str(out)])


def write_ring_tone(seconds: float, out: Path) -> None:
    """Узкополосный дребезг: так шум перестаёт быть «ровным белым» и мешает сильнее."""
    ffmpeg(["-f", "lavfi",
            "-i", f"sine=frequency={TONE_FREQ_HZ}:sample_rate={SAMPLE_RATE_HZ}:duration={seconds}",
            "-af", "vibrato=f=6:d=0.6,tremolo=f=3.5:d=0.7", "-ac", "1", str(out)])


def write_at_level(source: Path, target_mean_db: float, out: Path) -> None:
    gain_db = target_mean_db - mean_volume_db(source)
    ffmpeg(["-i", str(source), "-af", f"volume={gain_db}dB", str(out)])


def write_padded_speech(master: Path, total_s: float, lead_pad_s: float, out: Path) -> None:
    """Речь с паузами в начале и конце: шум должен звучать и там, где речи нет."""
    ffmpeg(["-i", str(master), "-ac", "1", "-ar", str(SAMPLE_RATE_HZ),
            "-af", f"adelay={int(lead_pad_s * 1000)},apad=whole_dur={total_s}", str(out)])


def write_mix(tracks: list[Path], out: Path) -> None:
    inputs: list[str] = []
    for track in tracks:
        inputs += ["-i", str(track)]
    chain = "".join(f"[{index}:a]" for index in range(len(tracks)))
    ffmpeg([*inputs, "-filter_complex",
            f"{chain}amix=inputs={len(tracks)}:duration=longest:normalize=0[out]",
            "-map", "[out]", str(out)])


def write_mp3(source: Path, out: Path) -> None:
    peak_db = peak_volume_db(source)
    if peak_db > PEAK_CEILING_DB:
        trimmed = source.with_name(source.stem + "_trimmed.wav")
        ffmpeg(["-i", str(source), "-af", f"volume=-{peak_db - PEAK_CEILING_DB}dB", str(trimmed)])
        source = trimmed
    ffmpeg(["-i", str(source), "-ar", str(SAMPLE_RATE_HZ), "-b:a", MP3_BITRATE, str(out)])


def build_case(case: NoiseCase, work_dir: Path) -> Path:
    master = SET_DIR / case.master
    speech_mean_db = mean_volume_db(master)
    total_s = duration_s(master) + case.lead_pad_s + case.tail_pad_s
    noise_mean_db = speech_mean_db - case.snr_db

    speech = work_dir / f"{case.out_id}_speech.wav"
    write_padded_speech(master, total_s, case.lead_pad_s, speech)

    noise_raw = work_dir / f"{case.out_id}_noise_raw.wav"
    write_white_noise(total_s, case.seed, noise_raw)
    noise = work_dir / f"{case.out_id}_noise.wav"
    write_at_level(noise_raw, noise_mean_db, noise)

    tracks = [speech, noise]
    if case.tone_below_noise_db is not None:
        tone_raw = work_dir / f"{case.out_id}_tone_raw.wav"
        write_ring_tone(total_s, tone_raw)
        tone = work_dir / f"{case.out_id}_tone.wav"
        write_at_level(tone_raw, noise_mean_db - case.tone_below_noise_db, tone)
        tracks.append(tone)

    mixed = work_dir / f"{case.out_id}_mixed.wav"
    write_mix(tracks, mixed)
    out = SET_DIR / f"{case.out_id}.mp3"
    write_mp3(mixed, out)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="показать рецепты и выйти")
    parser.add_argument("--case", action="append", default=[],
                        help="собрать только эти кейсы (по out_id), можно повторять")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print(f"{'case':16} {'master':18} {'SNR dB':>7} {'tone':>6} {'level':10} seed")
        for case in CASES:
            tone = "—" if case.tone_below_noise_db is None else f"-{case.tone_below_noise_db:g}"
            print(f"{case.out_id:16} {case.master:18} {case.snr_db:>7g} {tone:>6} "
                  f"{case.noise_level:10} {case.seed}")
        return 0

    selected = [case for case in CASES if not args.case or case.out_id in args.case]
    if not selected:
        raise SystemExit(f"нет таких кейсов: {', '.join(args.case)}")

    with tempfile.TemporaryDirectory(prefix="mix_noise_") as tmp:
        for case in selected:
            out = build_case(case, Path(tmp))
            print(f"{out.relative_to(ROOT)}: {duration_s(out):.2f}s "
                  f"mean={mean_volume_db(out):.1f}dB SNR={case.snr_db:g}dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
