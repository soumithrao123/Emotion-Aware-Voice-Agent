"""
test_emotion_predictor.py
Check EmotionPredictor on raw audio against results/metrics/clip_probabilities.csv.
Picks 5 test + 5 val clips (seed 42), runs predict_file on each wav, compares probabilities.
Pass: max abs diff < 1e-4 on every column.
"""
import time
import numpy as np
import pandas as pd

from emotion_predictor import EmotionPredictor, CODE_DIR

PROBS_CSV = CODE_DIR / "results" / "metrics" / "clip_probabilities.csv"
MANIFEST = CODE_DIR / "data" / "manifests" / "manifest.csv"
SEED, N_PER_SPLIT, TOL = 42, 5, 1e-4
COLS = ["p_angry", "p_happy", "p_sad", "p_distress"]

probs = pd.read_csv(PROBS_CSV)
manifest = pd.read_csv(MANIFEST, usecols=["utt_id", "wav_path"])
sample = pd.concat([
    probs[probs["split"] == s].sample(N_PER_SPLIT, random_state=SEED) for s in ["test", "val"]
]).merge(manifest, on="utt_id")

predictor = EmotionPredictor()
classes = predictor.classes

rows, times = [], []
for r in sample.itertuples():
    t0 = time.perf_counter()
    out = predictor.predict_file(CODE_DIR / r.wav_path)
    times.append(time.perf_counter() - t0)
    ref_label = classes[int(np.argmax([getattr(r, f"p_{c}") for c in classes]))]
    rows.append({
        "utt_id": r.utt_id, "split": r.split, "true": r.label,
        "pred": out["label"], "csv_pred": ref_label,
        **{f"d_{c}": abs(out["probs"][c] - getattr(r, f"p_{c}")) for c in classes},
        "d_distress": abs(out["p_distress"] - r.p_distress),
    })

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res[["utt_id", "split", "true", "pred", "csv_pred"]].to_string(index=False))

print("\nMax |diff| vs clip_probabilities.csv:")
max_diff = {}
for c in COLS:
    max_diff[c] = float(res["d_" + c.removeprefix("p_")].max())
    print(f"  {c:11s}: {max_diff[c]:.2e}")

labels_match = bool((res["pred"] == res["csv_pred"]).all())
print(f"\nPredicted labels match CSV argmax: {labels_match} ({(res['pred'] == res['csv_pred']).sum()}/{len(res)})")
print(f"Matches true label:               {(res['pred'] == res['true']).sum()}/{len(res)}")
print(f"Avg seconds per predict_file:     {np.mean(times):.2f}  (first call {times[0]:.2f}, min {min(times):.2f}, max {max(times):.2f})")

passed = max(max_diff.values()) < TOL and labels_match
print(f"\n{'PASS' if passed else 'FAIL'}: max diff {max(max_diff.values()):.2e} (tolerance {TOL})")
if not passed:
    raise SystemExit(1)
