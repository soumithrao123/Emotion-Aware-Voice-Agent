"""
record.py - record a short clip from the default microphone.
Saves a 16 kHz mono WAV to Code/data/my_voice/<name>.wav by default (never overwrites).
Usage:  python scripts/record.py 01_calm --seconds 5 --play
        python scripts/record.py 01_sad --out data/my_voice_v2      # fresh held-out set (Rule B)
--out is relative to Code/ unless absolute; the folder is created if missing.
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

TARGET_SR = 16000   # same sample rate the emotion model and Whisper use
CODE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = CODE_DIR / "data" / "my_voice"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name", help="file name without .wav, e.g. 01_calm")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--out", default=None,
                   help="output folder, relative to Code/ or absolute (default data/my_voice)")
    p.add_argument("--play", action="store_true", help="play the clip back after recording")
    args = p.parse_args()

    out_dir = OUT_DIR if args.out is None else Path(args.out)
    if not out_dir.is_absolute():
        out_dir = CODE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.name}.wav"
    if out.exists():
        raise SystemExit(f"{out.name} already exists. Pick a new name (nothing was overwritten).")

    # Record at the mic's own native rate (always supported), then resample to 16 kHz.
    mic_sr = int(sd.query_devices(kind="input")["default_samplerate"])
    input(f"Press Enter, then speak for {args.seconds:.0f} seconds...")
    print("Recording...")
    audio = sd.rec(int(args.seconds * mic_sr), samplerate=mic_sr, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    if mic_sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=mic_sr, target_sr=TARGET_SR)

    sf.write(out, audio, TARGET_SR)
    peak = float(np.abs(audio).max())
    print(f"Saved: {out}")
    print(f"Peak level: {peak:.2f}  (good range is about 0.1 to 0.9)")
    if peak < 0.02:
        print("WARNING: almost silent. Check the Windows mic permission or speak closer.")
    elif peak > 0.99:
        print("WARNING: clipping (too loud). Move a bit further from the mic.")

    if args.play:
        print("Playing back...")
        sd.play(audio, TARGET_SR)
        sd.wait()


if __name__ == "__main__":
    main()