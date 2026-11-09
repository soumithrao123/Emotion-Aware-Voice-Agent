"""
plot_mood_real.py
MoodManager vs per-turn reaction on REAL XGBoost outputs (held-out RAVDESS test speakers).
- Label script fixed in advance; clips drawn at random (seed 42) from ONE test speaker per conversation
- Parameters fixed in advance (gamma 0.7, threshold 0.5, persistence 2) - no tuning on test data
- 1,000 random conversations for aggregate numbers; the first one drawn is plotted
Reads:  results/metrics/clip_probabilities.csv
Writes: results/figures/mood_trajectory_real.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mood_manager import MoodManager, reactive_mode

PROBS_CSV = Path("results/metrics/clip_probabilities.csv")
OUT_PNG = Path("results/figures/mood_trajectory_real.png")
SEED, N_RUNS = 42, 1000
GAMMA, THRESHOLD, PERSISTENCE = 0.7, 0.5, 2
LABEL_SCRIPT = ["happy", "happy", "angry", "happy", "sad", "sad",
                "angry", "happy", "sad", "happy", "happy", "happy"]

probs = pd.read_csv(PROBS_CSV)
test = probs[probs["split"] == "test"]
speakers = sorted(int(s) for s in test["speaker_id"].unique())


def build_sequence(speaker_id, rng):
    clips = test[test["speaker_id"] == speaker_id]
    decks = {e: clips[clips["label"] == e].sample(frac=1, random_state=rng).to_dict("records")
             for e in ["angry", "happy", "sad"]}
    return pd.DataFrame([decks[e].pop() for e in LABEL_SCRIPT])


def run_agents(p_seq):
    mgr = MoodManager(GAMMA, THRESHOLD, PERSISTENCE)
    scores, mood, react = [], [], []
    for p in p_seq:
        mood.append(mgr.update(p))
        scores.append(mgr.score)
        react.append(reactive_mode(p, THRESHOLD))
    return scores, mood, react


def n_switches(modes):
    return sum(a != b for a, b in zip(modes, modes[1:]))


# ---- 1,000 random conversations ----
rng = np.random.default_rng(SEED)
rows, example = [], None
for i in range(N_RUNS):
    spk = rng.choice(speakers)
    seq = build_sequence(spk, rng)
    scores, mood, react = run_agents(seq["p_distress"].tolist())
    if example is None:                       # plot the FIRST draw, not a hand-picked one
        example = (spk, seq, scores, mood, react)
    rows.append({
        "react_switches": n_switches(react),
        "mood_switches": n_switches(mood),
        "react_spike": react[2] == "EMPATHETIC",             # turn 3 = one-off angry turn
        "mood_spike": mood[2] == "EMPATHETIC",
        "mood_detects_distress": "EMPATHETIC" in mood[4:9],  # turns 5-9
        "mood_recovers": mood[-1] == "DEFAULT",              # back to default by turn 12
    })
r = pd.DataFrame(rows)

print(f"=== {N_RUNS} random conversations, test speakers {speakers} ===")
print(f"Mode switches   reactive {r.react_switches.mean():.2f} ± {r.react_switches.std():.2f} | "
      f"MoodManager {r.mood_switches.mean():.2f} ± {r.mood_switches.std():.2f}")
print(f"Switched on one-off spike (turn 3)   reactive {r.react_spike.mean():.1%} | MoodManager {r.mood_spike.mean():.1%}")
print(f"MoodManager reached EMPATHETIC during sustained distress: {r.mood_detects_distress.mean():.1%}")
print(f"MoodManager back to DEFAULT by turn 12: {r.mood_recovers.mean():.1%}")

# ---- Plot the example conversation ----
spk, seq, scores, mood, react = example
p = seq["p_distress"].tolist()
turns = list(range(1, len(p) + 1))
GREEN, LIGHT, GREY, EMP = "#00853E", "#A8D5BA", "#9E9E9E", "#E07A5F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16})

fig, (ax, axm) = plt.subplots(2, 1, figsize=(12, 8.4), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08})
ax.plot(turns, p, "o--", color=GREY, ms=10, lw=1.5, alpha=0.85,
        label="Classifier output per turn (XGBoost)")
ax.plot(turns, scores, "-", color=GREEN, lw=4.5, marker="o", ms=9,
        label=f"MoodManager score (EMA, γ = {GAMMA})")
ax.axhline(THRESHOLD, color="black", ls=":", lw=2)
ax.text(12.6, THRESHOLD, f"threshold\n= {THRESHOLD}", va="center", fontsize=13)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Distress score\n(P(angry) + P(sad))", fontsize=17)
ax.set_title(f"Real classifier outputs, held-out test speaker (Actor {spk})", fontsize=20, pad=14)
ax.legend(loc="upper left", fontsize=13, frameon=False)
ax.grid(axis="y", alpha=0.3)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

for i, (name, modes) in enumerate([("Reactive", react), ("MoodManager", mood)]):
    y = 1 - i
    for t, m in zip(turns, modes):
        axm.barh(y, 1, left=t - 0.5, height=0.7,
                 color=EMP if m == "EMPATHETIC" else LIGHT, edgecolor="white", lw=2)
    axm.text(12.6, y, f"{n_switches(modes)} switch" + ("" if n_switches(modes) == 1 else "es"), va="center", fontsize=14)
axm.set_yticks([1, 0])
axm.set_yticklabels(["Reactive", "MoodManager"], fontsize=15)
axm.set_xticks(turns)
axm.set_xticklabels([f"{t}\n{lab}" for t, lab in zip(turns, seq["label"])], fontsize=12)
axm.set_xlabel("Conversation turn (true emotion of the clip)", fontsize=16)
axm.set_xlim(0.5, 12.5)
for s in ["top", "right", "left"]:
    axm.spines[s].set_visible(False)
axm.tick_params(axis="y", length=0)
axm.legend(handles=[Patch(color=LIGHT, label="DEFAULT"), Patch(color=EMP, label="EMPATHETIC")],
           loc="upper center", bbox_to_anchor=(0.5, -0.75), ncol=2, fontsize=13, frameon=False)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print("Saved ->", OUT_PNG)
print(seq[["label", "utt_id", "p_distress"]].assign(score=np.round(scores, 3), react=react, mood=mood)
      .round(3).to_string())