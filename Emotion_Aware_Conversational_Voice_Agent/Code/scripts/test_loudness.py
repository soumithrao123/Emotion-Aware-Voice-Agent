"""
test_loudness.py
Does loudness explain why my_voice_v2 reads as angry?

For each clip in data/my_voice (01-10) and data/my_voice_v2 (01-15), in memory only (no audio is written):
  speech RMS = RMS of the clip after the same silence trim the predictor uses (preprocess)
  gain       = target / speech RMS, target = median speech RMS of the RAVDESS train clips
  run the current XGBoost model on the clip before and after scaling.
The trim is relative to the clip's own peak (top_db), so scaling doesn't change what gets trimmed.

Reports, per group and before/after: accuracy, prediction counts, and z-scores of mfcc1_mean / rms_mean /
rms_std vs RAVDESS train (manifest features, the data the model was fit on).
Outputs: results/metrics/loudness_test.csv (one row per clip x condition)
Usage: python scripts/test_loudness.py
"""
import re

import numpy as np
import pandas as pd

from feature_extraction import load_audio, preprocess, extract_features
from emotion_predictor import EmotionPredictor, CODE_DIR

# Config
MANIFEST = CODE_DIR / "data" / "manifests" / "manifest.csv"
OUT_CSV = CODE_DIR / "results" / "metrics" / "loudness_test.csv"
GROUPS = {
    "my_voice": (CODE_DIR / "data" / "my_voice", range(1, 11)),
    "my_voice_v2": (CODE_DIR / "data" / "my_voice_v2", range(1, 16)),
}
Z_FEATURES = ["mfcc1_mean", "rms_mean", "rms_std"]
CLASSES = ["angry", "happy", "sad"]


def speech_rms(y, sr):
    t = preprocess(y, sr)
    return float(np.sqrt(np.mean(t ** 2)))


def clips(folder, numbers):
    for p in sorted(folder.glob("*.wav")):
        m = re.fullmatch(r"(\d+)_(happy|sad|angry)", p.stem)
        if m and int(m.group(1)) in numbers:
            yield p, m.group(2)


def main():
    man = pd.read_csv(MANIFEST)
    train = man[man["split"] == "train"]

    print(f"Speech RMS of {len(train)} RAVDESS train clips...", flush=True)
    train_rms = np.array([speech_rms(*load_audio(CODE_DIR / w)) for w in train["wav_path"]])
    target = float(np.median(train_rms))
    print(f"  median {target:.4f} ({20 * np.log10(target):.1f} dBFS); "
          f"IQR {np.percentile(train_rms, 25):.4f}-{np.percentile(train_rms, 75):.4f}")

    pred = EmotionPredictor()
    rows = []
    for group, (folder, nums) in GROUPS.items():
        for path, label in clips(folder, nums):
            y, sr = load_audio(path)
            rms_before = speech_rms(y, sr)
            gain = target / rms_before
            for cond, audio in [("before", y), ("after", (y * gain).astype(np.float32))]:
                feats = extract_features(preprocess(audio, sr), sr)
                row = pd.DataFrame([feats])[pred.feature_cols]
                proba = pred.model.predict_proba(row)[0]
                rows.append({"group": group, "file": path.name, "label": label, "condition": cond,
                             "speech_rms": speech_rms(audio, sr), "gain_db": 20 * np.log10(gain),
                             "peak": float(np.abs(audio).max()),
                             "pred": CLASSES[int(np.argmax(proba))], "confidence": float(proba.max()),
                             **{f"p_{c}": float(p) for c, p in zip(CLASSES, proba)},
                             **{f: feats[f] for f in Z_FEATURES}})
            print(f"  {group:12s} {path.name:14s} gain {20 * np.log10(gain):+5.1f} dB  "
                  f"{rows[-2]['pred']:5s} -> {rows[-1]['pred']}", flush=True)
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.5g")

    mu, sd = train[Z_FEATURES].mean(), train[Z_FEATURES].std()
    print("\nRAVDESS train class means (z):")
    for c in CLASSES:
        z = (train[train["label"] == c][Z_FEATURES].mean() - mu) / sd
        print(f"  {c:5s} " + "  ".join(f"{f}={z[f]:+.2f}" for f in Z_FEATURES))

    print(f"\n{'group':12s} {'cond':6s} {'acc':>5s}  {'pred counts':24s} "
          + "  ".join(f"{('z_' + f):>12s}" for f in Z_FEATURES) + "   per-label correct")
    for group in GROUPS:
        for cond in ["before", "after"]:
            g = df[(df["group"] == group) & (df["condition"] == cond)]
            acc = (g["pred"] == g["label"]).mean()
            counts = " ".join(f"{c}:{(g['pred'] == c).sum()}" for c in CLASSES)
            z = (g[Z_FEATURES].mean() - mu) / sd
            per = " ".join(f"{c} {((g['label'] == c) & (g['pred'] == c)).sum()}/{(g['label'] == c).sum()}"
                           for c in ["happy", "sad", "angry"])
            print(f"{group:12s} {cond:6s} {acc:5.2f}  {counts:24s} "
                  + "  ".join(f"{z[f]:12.2f}" for f in Z_FEATURES) + f"   {per}")
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
