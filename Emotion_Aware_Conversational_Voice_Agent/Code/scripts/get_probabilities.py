"""
get_probabilities.py
Export per-clip emotion probabilities from XGBoost for held-out clips.
"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Config
MANIFEST = Path("data/manifests/manifest.csv")
SEED = 42

df = pd.read_csv(MANIFEST)
meta_cols = ["utt_id", "speaker_id", "wav_path", "label", "split", "duration"]
feature_cols = [c for c in df.columns if c not in meta_cols]

train_df = df[df["split"] == "train"]
val_df   = df[df["split"] == "val"]
test_df  = df[df["split"] == "test"]

print("Features:", len(feature_cols))
print("Train:", len(train_df), " Val:", len(val_df), " Test:", len(test_df))

# Encode text labels as numbers (XGBoost needs 0, 1, 2)

le = LabelEncoder()
y_train = le.fit_transform(train_df["label"])
print("Class order:", dict(enumerate(le.classes_)))

# Same settings as train_baseline.py
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

# match with the xgboost predections

from sklearn.metrics import accuracy_score, f1_score

# Predict on the test set and turn numbers back into labels
y_pred = le.inverse_transform(xgb.predict(test_df[feature_cols]))

acc = accuracy_score(test_df["label"], y_pred)
f1 = f1_score(test_df["label"], y_pred, average="macro")

print(f"Test accuracy: {acc:.3f}  (poster: 0.729)")
print(f"Test macro-F1: {f1:.3f}  (poster: 0.715)")

#fetch probabilities and save them
# Take ONLY held-out clips (val + test), never training clips
held = df[df["split"].isin(["val", "test"])].copy()

# Probability for each emotion, per clip -> array of shape (192, 3)
proba = xgb.predict_proba(held[feature_cols])

# One column per emotion, named from the encoder's class order
for i, emotion in enumerate(le.classes_):
    held[f"p_{emotion}"] = proba[:, i]

# The single "distress" score MoodManager will track
held["p_distress"] = held["p_angry"] + held["p_sad"]

# Keep only what we need and save
out_cols = ["utt_id", "speaker_id", "split", "label",
            "p_angry", "p_happy", "p_sad", "p_distress"]
OUT_CSV = Path("results/metrics/clip_probabilities.csv")
held[out_cols].to_csv(OUT_CSV, index=False)

print(f"\nSaved {len(held)} clips -> {OUT_CSV}")
print(held[out_cols].head())