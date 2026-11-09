"""
check_audio.py
Why does my_voice_v2 read as angry? Compare three groups of clips:
  my_voice    data/my_voice/01-10   (recorded 2026-10-31)
  my_voice_v2 data/my_voice_v2/01-15 (recorded 2026-11-05)
  ravdess_test  the 96 RAVDESS test clips (manifest split == "test")
Reference = RAVDESS train (manifest split == "train"), the data the XGBoost model was fit on.

Per clip: file header (sample rate, channels, subtype), duration, peak, RMS (dBFS), % clipped samples
(|x| >= 0.99), noise floor (10th-percentile frame RMS, dBFS), trimmed duration, the 39 features computed
exactly as EmotionPredictor does (load_audio -> preprocess -> extract_features), and the prediction.
Then: z-score of each group's mean vs RAVDESS train for the top-10 XGBoost features (by gain).

Outputs: results/metrics/audio_check.csv (per clip + one MEAN row per group),
         results/metrics/audio_check_zscores.csv (top-10 features x groups)
Usage: python scripts/check_audio.py
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

from feature_extraction import load_audio, preprocess, extract_features
from emotion_predictor import EmotionPredictor, CODE_DIR

# Config
MANIFEST = CODE_DIR / "data" / "manifests" / "manifest.csv"
OUT_CSV = CODE_DIR / "results" / "metrics" / "audio_check.csv"
OUT_Z = CODE_DIR / "results" / "metrics" / "audio_check_zscores.csv"
CLIP_LEVEL = 0.99
TOP_K = 10
MY_GROUPS = {
    "my_voice": (CODE_DIR / "data" / "my_voice", range(1, 11)),
    "my_voice_v2": (CODE_DIR / "data" / "my_voice_v2", range(1, 16)),
}


def my_clips(folder, numbers):
    """NN_label.wav files whose NN is in `numbers`, sorted by name (skips 00_test*)."""
    out = []
    for p in sorted(folder.glob("*.wav")):
        m = re.fullmatch(r"(\d+)_(happy|sad|angry)", p.stem)
        if m and int(m.group(1)) in numbers:
            out.append((p, m.group(2)))
    return out


def db(x):
    return float(20 * np.log10(max(x, 1e-10)))


def audio_stats(path):
    info = sf.info(str(path))
    raw, _ = sf.read(str(path), dtype="float32", always_2d=True)   # file as stored, before any resampling
    mono = raw.mean(axis=1)
    y, sr = load_audio(path)                                         # what the predictor sees (16 kHz mono)
    frame_rms = librosa.feature.rms(y=y)[0]
    return y, sr, {
        "file_sr": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "duration_s": round(info.duration, 3),
        "peak": float(np.abs(mono).max()),
        "rms_dbfs": db(np.sqrt(np.mean(mono ** 2))),
        "pct_clipped": float(100 * np.mean(np.abs(raw) >= CLIP_LEVEL)),
        "noise_floor_dbfs": db(np.percentile(frame_rms, 10)),
        "trimmed_s": round(preprocess(y, sr).size / sr, 3),
    }


def main():
    pred = EmotionPredictor()
    feat_cols = pred.feature_cols
    man = pd.read_csv(MANIFEST)
    train = man[man["split"] == "train"]

    jobs = [(g, p, lab) for g, (folder, nums) in MY_GROUPS.items() for p, lab in my_clips(folder, nums)]
    jobs += [("ravdess_test", CODE_DIR / r.wav_path, r.label) for r in man[man["split"] == "test"].itertuples()]

    rows = []
    for i, (group, path, label) in enumerate(jobs, 1):
        y, sr, stats = audio_stats(path)
        feats = extract_features(preprocess(y, sr), sr)
        out = pred.predict_array(y, sr)
        rows.append({"group": group, "file": path.name, "label": label, **stats,
                     "pred": out["label"], "confidence": max(out["probs"].values()),
                     "p_distress": out["p_distress"], **feats})
        print(f"\r{i}/{len(jobs)} {group:12s} {path.name}", end="", flush=True)
    print()
    df = pd.DataFrame(rows)

    # Sanity: recomputed RAVDESS test features should match the manifest the model was trained alongside
    rt = df[df["group"] == "ravdess_test"].set_index("file")[feat_cols]
    mt = man[man["split"] == "test"].assign(file=lambda d: d["utt_id"] + ".wav").set_index("file")[feat_cols]
    max_diff = float((rt - mt.loc[rt.index]).abs().max().max())
    print(f"RAVDESS test: recomputed vs manifest features, max abs diff = {max_diff:.2e}")

    num_cols = [c for c in df.columns if c not in ("group", "file", "label", "subtype", "pred")]
    means = df.groupby("group", sort=False)[num_cols].mean().reset_index()
    means["file"] = "MEAN"
    means["pred"] = df.groupby("group", sort=False)["pred"].agg(
        lambda s: " ".join(f"{k}:{v}" for k, v in s.value_counts().items())).values
    means["label"] = df.groupby("group", sort=False).apply(
        lambda g: f"acc {(g['pred'] == g['label']).mean():.2f}", include_groups=False).values
    out = pd.concat([df, means], ignore_index=True)[df.columns]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.5g")

    # Top-10 features by XGBoost gain; z = (group mean - train mean) / train std
    gain = pred.model.get_booster().get_score(importance_type="gain")
    imp = pd.Series({c: gain.get(c, gain.get(f"f{i}", 0.0)) for i, c in enumerate(feat_cols)})
    top = imp.sort_values(ascending=False).head(TOP_K).index
    mu, sd = train[top].mean(), train[top].std()
    ztab = pd.DataFrame({"gain_rank": range(1, TOP_K + 1), "train_mean": mu.values, "train_std": sd.values},
                        index=top)
    for g in ["my_voice", "my_voice_v2", "ravdess_test"]:
        ztab[f"mean_{g}"] = df[df["group"] == g][top].mean().values
        ztab[f"z_{g}"] = ((ztab[f"mean_{g}"] - mu) / sd).values
    ztab["z_v2_minus_v1"] = ztab["z_my_voice_v2"] - ztab["z_my_voice"]
    ztab.index.name = "feature"
    ztab.to_csv(OUT_Z, float_format="%.4g")

    # Report
    pd.set_option("display.width", 200)
    show = ["file_sr", "channels", "duration_s", "trimmed_s", "peak", "rms_dbfs", "noise_floor_dbfs",
            "pct_clipped", "confidence", "p_distress"]
    print("\nGroup averages (audio):")
    print(means.set_index("group")[show + ["label", "pred"]].round(3).to_string())
    print("\nHeader formats:", df.groupby("group")[["file_sr", "channels", "subtype"]]
          .agg(lambda s: sorted(set(s))).to_string())
    print(f"\nTop-{TOP_K} features: z-score of group mean vs RAVDESS train")
    print(ztab[["gain_rank", "z_my_voice", "z_my_voice_v2", "z_ravdess_test", "z_v2_minus_v1"]].round(2).to_string())
    print(f"\nSaved {OUT_CSV}\nSaved {OUT_Z}")


if __name__ == "__main__":
    main()
