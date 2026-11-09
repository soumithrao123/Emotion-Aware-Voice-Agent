"""
test_asr.py
Check ASR (whisper tiny.en) on the same 10 clips as test_emotion_predictor.py.
Expected text comes from RAVDESS filename field 5 (statement code):
  "01" = "Kids are talking by the door", "02" = "Dogs are sitting by the door"
Comparison is on normalized text (lowercase, punctuation stripped).
"""
import re
import time
import numpy as np
import pandas as pd

from asr import ASR
from emotion_predictor import CODE_DIR

PROBS_CSV = CODE_DIR / "results" / "metrics" / "clip_probabilities.csv"
MANIFEST = CODE_DIR / "data" / "manifests" / "manifest.csv"
SEED, N_PER_SPLIT = 42, 5
STATEMENTS = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", text.lower()).split())


def wer(ref: str, hyp: str) -> float:
    """Word error rate: word-level edit distance / number of reference words."""
    r, h = ref.split(), hyp.split()
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r)


# Same 10 clips as test_emotion_predictor.py
probs = pd.read_csv(PROBS_CSV)
manifest = pd.read_csv(MANIFEST, usecols=["utt_id", "wav_path"])
sample = pd.concat([
    probs[probs["split"] == s].sample(N_PER_SPLIT, random_state=SEED) for s in ["test", "val"]
]).merge(manifest, on="utt_id")

t0 = time.perf_counter()
asr = ASR()
load_time = time.perf_counter() - t0

rows, times = [], []
for r in sample.itertuples():
    expected = STATEMENTS[r.utt_id.split("-")[4]]
    t0 = time.perf_counter()
    text = asr.transcribe_file(CODE_DIR / r.wav_path)
    times.append(time.perf_counter() - t0)
    rows.append({
        "utt_id": r.utt_id,
        "expected": expected,
        "transcript": text,
        "match": normalize(text) == normalize(expected),
        "wer": round(wer(normalize(expected), normalize(text)), 3),
        "sec": round(times[-1], 2),
    })

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 40)
print(res.to_string(index=False))

print(f"\nExact matches (normalized): {res['match'].sum()}/{len(res)}")
print(f"Mean WER:                   {res['wer'].mean():.3f}")
print(f"Model load time:            {load_time:.2f} s")
print(f"Avg seconds per transcribe: {np.mean(times):.2f}  (first call {times[0]:.2f}, min {min(times):.2f}, max {max(times):.2f})")
