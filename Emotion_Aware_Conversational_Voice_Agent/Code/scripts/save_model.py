"""
save_model.py
Train the XGBoost baseline (same data/settings as get_probabilities.py) and save it to disk.
Writes: checkpoints/xgb_baseline.json  (model)
        checkpoints/xgb_meta.json      (class order, feature columns, preprocessing settings)
Then reloads the saved model and checks it against results/metrics/clip_probabilities.csv.
"""
from pathlib import Path
from datetime import date
import json
import numpy as np
import pandas as pd
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# Preprocessing settings come from the script that built the features
from feature_extraction import SAMPLE_RATE, TRIM_TOP_DB, N_MFCC, USE_PITCH

# Config
MANIFEST = Path("data/manifests/manifest.csv")
PROBS_CSV = Path("results/metrics/clip_probabilities.csv")
CKPT_DIR = Path("checkpoints")
MODEL_PATH = CKPT_DIR / "xgb_baseline.json"
META_PATH = CKPT_DIR / "xgb_meta.json"
SEED = 42
MIN_LENGTH = 512   # feature_extraction.main(): if trimmed clip < 512 samples, use the untrimmed one
TOL = 1e-5

df = pd.read_csv(MANIFEST)
meta_cols = ["utt_id", "speaker_id", "wav_path", "label", "split", "duration"]
feature_cols = [c for c in df.columns if c not in meta_cols]

train_df = df[df["split"] == "train"]
test_df  = df[df["split"] == "test"]

print("Features:", len(feature_cols))
print("Train:", len(train_df), " Test:", len(test_df))

# Encode text labels as numbers (XGBoost needs 0, 1, 2)
le = LabelEncoder()
y_train = le.fit_transform(train_df["label"])
print("Class order:", dict(enumerate(le.classes_)))

# Same settings as get_probabilities.py / train_baseline.py
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED,
    n_jobs=-1,
    eval_metric="mlogloss",
)
xgb.fit(train_df[feature_cols], y_train)
print("XGBoost trained.")

# Save model + metadata
CKPT_DIR.mkdir(parents=True, exist_ok=True)
xgb.save_model(MODEL_PATH)

meta = {
    "classes": [str(c) for c in le.classes_],
    "feature_cols": feature_cols,
    "preprocessing": {
        "SAMPLE_RATE": SAMPLE_RATE,
        "TRIM_TOP_DB": TRIM_TOP_DB,
        "N_MFCC": N_MFCC,
        "USE_PITCH": USE_PITCH,
        "MIN_LENGTH_FALLBACK": MIN_LENGTH,
    },
    "xgboost_version": xgboost.__version__,
    "trained_on": date.today().isoformat(),
}
META_PATH.write_text(json.dumps(meta, indent=2))
print(f"Saved -> {MODEL_PATH}")
print(f"Saved -> {META_PATH}")

# Verify: reload from disk into a fresh model and compare to the exported probabilities
loaded = XGBClassifier()
loaded.load_model(MODEL_PATH)
meta_loaded = json.loads(META_PATH.read_text())
classes = meta_loaded["classes"]
cols = meta_loaded["feature_cols"]

held = df[df["split"].isin(["val", "test"])].copy()
proba = loaded.predict_proba(held[cols])
for i, emotion in enumerate(classes):
    held[f"p_{emotion}"] = proba[:, i]
held["p_distress"] = held["p_angry"] + held["p_sad"]

ref = pd.read_csv(PROBS_CSV)
merged = held.merge(ref, on="utt_id", suffixes=("_new", "_ref"))
print(f"\nVerification: {len(merged)} of {len(held)} held-out clips matched by utt_id")

prob_cols = [f"p_{e}" for e in classes] + ["p_distress"]
max_diff = {}
for c in prob_cols:
    max_diff[c] = float(np.max(np.abs(merged[f"{c}_new"] - merged[f"{c}_ref"])))
    print(f"  max |diff| {c:11s}: {max_diff[c]:.2e}")

test_rows = held[held["split"] == "test"]
y_pred = [classes[i] for i in np.argmax(proba[(held["split"] == "test").values], axis=1)]
acc = accuracy_score(test_rows["label"], y_pred)
f1 = f1_score(test_rows["label"], y_pred, average="macro")
print(f"\nTest accuracy: {acc:.3f}")
print(f"Test macro-F1: {f1:.3f}")

if len(merged) != len(held) or max(max_diff.values()) > TOL:
    raise SystemExit(f"\nMISMATCH: probabilities differ from {PROBS_CSV} (tolerance {TOL}).")
print(f"\nOK: saved model reproduces {PROBS_CSV} within {TOL}.")
