"""
agent.py
Task A: thin end-to-end agent, printed to the terminal (no TTS, no memory).
Per turn: audio -> emotion probs + transcript -> MoodManager mode -> mood-steered reply.
Logs every turn as one JSON line to results/logs/session_<timestamp>.jsonl.

Usage (from anywhere):
  python scripts/agent.py clip1.wav clip2.wav ...
  python scripts/agent.py data/my_voice_v2           # a folder = its .wav files, sorted by name
  python scripts/agent.py --script            # 12-turn label script from one test speaker (seed 42)
  flags: --no-llm (template replies only), --no-mood (per-turn reactive baseline),
         --tau T (abstain below top-1 confidence T; default TAU_CONF from abstain_curve.py, 0 disables)
"""
from collections import deque
from datetime import datetime
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd

from feature_extraction import load_audio, SAMPLE_RATE
from emotion_predictor import EmotionPredictor, CODE_DIR
from asr import ASR
from mood_manager import MoodManager, reactive_mode, TAU_CONF
from response_generator import generate_reply, build_system_prompt

# Config
GAMMA, THRESHOLD, PERSISTENCE = 0.7, 0.5, 2   # MoodManager v1 defaults (same as plot_mood_real.py)
HISTORY_TURNS = 4
LOG_DIR = CODE_DIR / "results" / "logs"
MANIFEST = CODE_DIR / "data" / "manifests" / "manifest.csv"
SEED = 42
LABEL_SCRIPT = ["happy", "happy", "angry", "happy", "sad", "sad",
                "angry", "happy", "sad", "happy", "happy", "happy"]
# Phrased to read naturally in build_system_prompt's "from their voice, they seem {perceived_emotion}."
DISTRESS_CONTEXT = "to have sounded distressed over the last few turns"


class Agent:
    def __init__(self, use_llm=True, use_mood=True, tau_conf=TAU_CONF):
        t0 = time.perf_counter()
        self.tau_conf = tau_conf
        self.use_llm = use_llm
        self.use_mood = use_mood
        self.emotion = EmotionPredictor()
        self.asr = ASR()
        self.mood = MoodManager(GAMMA, THRESHOLD, PERSISTENCE)
        self.history = deque(maxlen=HISTORY_TURNS)   # (user_text, agent_text) pairs
        self.turn = 0
        self.prev_mode = None

        # Warm-up on 1.5 s of synthetic audio: pays the one-time JIT / model init cost now
        rng = np.random.default_rng(0)
        t = np.arange(int(1.5 * SAMPLE_RATE)) / SAMPLE_RATE
        y = (0.1 * np.sin(2 * np.pi * 180 * t) + 0.01 * rng.standard_normal(t.size)).astype(np.float32)
        self.emotion.predict_array(y, SAMPLE_RATE)
        self.asr.transcribe_array(y, SAMPLE_RATE)

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        self.startup_s = time.perf_counter() - t0
        print(f"Agent ready in {self.startup_s:.1f}s  (llm={use_llm}, mood={use_mood}, tau={tau_conf:.2f})  log -> {self.log_path}")

    def run_turn(self, wav_path) -> dict:
        self.turn += 1
        timings = {}
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        y, sr = load_audio(wav_path)          # loaded once, shared by emotion + ASR
        timings["load"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        emo = self.emotion.predict_array(y, sr)
        timings["emotion"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        transcript = self.asr.transcribe_array(y, sr)
        timings["asr"] = time.perf_counter() - t0

        confidence = max(emo["probs"].values())
        t0 = time.perf_counter()
        if self.use_mood:
            mode = self.mood.update(emo["p_distress"], confidence, self.tau_conf)
            mood_score, turns_above = self.mood.score, self.mood.turns_above
            abstained, check_in = self.mood.abstained, self.mood.check_in
            reason = self.mood.explain()
        else:
            mode = reactive_mode(emo["p_distress"], THRESHOLD)   # per-turn baseline for ablation
            mood_score, turns_above = None, None
            abstained, check_in = False, False
            reason = "reactive baseline: p_distress vs 0.5"
        timings["mood"] = time.perf_counter() - t0
        switched = self.prev_mode is not None and mode != self.prev_mode
        self.prev_mode = mode

        # Tone comes from the mood state only. The per-turn classifier label is noisy
        # (~72% accurate on held-out speakers); passing it to the LLM would let one
        # misclassified clip change the agent's tone and undo MoodManager's smoothing.
        perceived_emotion = DISTRESS_CONTEXT if mode == "EMPATHETIC" else None

        # Same arguments generate_reply passes to build_system_prompt, so this is the exact LLM prompt
        system_prompt = build_system_prompt(mode, perceived_emotion, None) if self.use_llm else None

        t0 = time.perf_counter()
        reply = generate_reply(mode, transcript, perceived_emotion=perceived_emotion,
                               memory_facts=None, history=list(self.history), use_llm=self.use_llm)
        timings["llm"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - t_start
        self.history.append((transcript, reply["text"]))

        result = {
            "turn": self.turn,
            "file": Path(wav_path).name,
            "transcript": transcript,
            "probs": emo["probs"],
            "label": emo["label"],
            "p_distress": emo["p_distress"],
            "mood_score": mood_score,
            "turns_above": turns_above,
            "mode": mode,
            "switched": switched,
            "perceived_emotion": perceived_emotion,
            "reply": reply["text"],
            "source": reply["source"],
            "model": reply["model"],
            "error": reply["error"],
            "attempts": reply["attempts"],
            "timings": {k: round(v, 3) for k, v in timings.items()},
            "use_llm": self.use_llm,
            "use_mood": self.use_mood,
            # A2.1 inspectable state
            "confidence": confidence,
            "mood_params": {"gamma": GAMMA, "threshold": THRESHOLD, "persistence": PERSISTENCE,
                            "tau_conf": self.tau_conf},
            "reason": reason,
            "system_prompt": system_prompt,
            # Placeholders for later A2 steps; always present so the log schema is stable
            "abstained": abstained,   # A2.2
            "check_in": check_in,     # A2.2 (logged only; used in the prompt from A2.3)
            "strategy": None,
            "tts_tags": None,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        self.print_turn(result)
        return result

    @staticmethod
    def print_turn(r):
        probs = "  ".join(f"{k}={v:.2f}" for k, v in r["probs"].items())
        mood = (f"score={r['mood_score']:.3f}  turns_above={r['turns_above']}"
                if r["mood_score"] is not None else "score=n/a (reactive)")
        t = r["timings"]
        print(f"\n--- Turn {r['turn']}: {r['file']} ---")
        print(f"  transcript : {r['transcript']}")
        print(f"  emotion    : {probs}  -> label={r['label']}  confidence={r['confidence']:.2f}  p_distress={r['p_distress']:.3f}")
        print(f"  mood       : {mood}  mode={r['mode']}{'  << SWITCHED' if r['switched'] else ''}")
        flags = "  ".join(f for f in ["ABSTAINED" if r["abstained"] else "",
                                      "CHECK_IN" if r["check_in"] else ""] if f)
        print(f"  reason     : {r['reason']}" + (f"   [{flags}]" if flags else ""))
        print(f"  reply      : {r['reply']}")
        src = f"{r['source']} ({r['model']})" if r["model"] else r["source"]
        print(f"  source     : {src}" + (f"  error={r['error']}" if r["error"] else ""))
        if fell_back(r):
            print(f"  fallback   : {chain(r)}")
        print(f"  timings    : load {t['load']:.2f}s | emotion {t['emotion']:.2f}s | asr {t['asr']:.2f}s | "
              f"mood {t['mood'] * 1000:.2f}ms | llm {t['llm']:.2f}s | total {t['total']:.2f}s")


def fell_back(r):
    a = r["attempts"]
    return len(a) > 1 or (len(a) == 1 and a[0]["outcome"] != "ok")


def chain(r):
    """e.g. '3.6-flash: 429 (0.4s) -> 3.5-flash-lite: ok (1.1s)'"""
    return " -> ".join(f"{a['model'].removeprefix('gemini-')}: {a['outcome']} ({a['seconds']:.1f}s)"
                       for a in r["attempts"])


def scripted_clips():
    """Same draw as plot_mood_real.py's first conversation: one test speaker, a shuffled deck per label."""
    df = pd.read_csv(MANIFEST, usecols=["utt_id", "speaker_id", "wav_path", "label", "split"])
    test = df[df["split"] == "test"]
    speakers = sorted(int(s) for s in test["speaker_id"].unique())
    rng = np.random.default_rng(SEED)
    speaker = int(rng.choice(speakers))
    clips = test[test["speaker_id"] == speaker]
    decks = {e: clips[clips["label"] == e].sample(frac=1, random_state=rng).to_dict("records")
             for e in ["angry", "happy", "sad"]}
    picked = [decks[e].pop() for e in LABEL_SCRIPT]
    return speaker, [CODE_DIR / c["wav_path"] for c in picked], [c["label"] for c in picked]


def summary(results, true_labels=None):
    modes = [r["mode"] for r in results]
    n_switches = sum(a != b for a, b in zip(modes, modes[1:]))
    print("\n" + "=" * 70)
    print(f"SUMMARY  ({len(results)} turns, mood={'MoodManager' if results[0]['use_mood'] else 'reactive'}, "
          f"llm={results[0]['use_llm']})")
    print("=" * 70)
    print(f"Mode switches: {n_switches}")
    abst = [r["turn"] for r in results if r["abstained"]]
    print(f"Abstained turns: {abst or 'none'}   check_in turns: "
          f"{[r['turn'] for r in results if r['check_in']] or 'none'}")
    for r in results:
        true = f"  true={true_labels[r['turn'] - 1]:5s}" if true_labels else ""
        model = f" ({r['model']})" if r["model"] else ""
        print(f"  turn {r['turn']:2d}{true}  pred={r['label']:5s}  p_distress={r['p_distress']:.2f}  "
              f"conf={r['confidence']:.2f}{'*' if r['abstained'] else ' '} mode={r['mode']:10s}  source={r['source']}{model}")
    totals = [r["timings"]["total"] for r in results]
    print(f"Average turn latency: {np.mean(totals):.2f}s  (min {min(totals):.2f}s, max {max(totals):.2f}s)")
    fallbacks = [r for r in results if fell_back(r)]
    print(f"Turns with model fallback: {len(fallbacks)}")
    for r in fallbacks:
        print(f"  turn {r['turn']:2d}: {chain(r)}")


def main():
    ap = argparse.ArgumentParser(description="Thin end-to-end emotion-aware agent (Task A).")
    ap.add_argument("clips", nargs="*", help="wav files to run in order, or folders (their .wav files, sorted)")
    ap.add_argument("--script", action="store_true", help="12-turn label script from one test speaker")
    ap.add_argument("--no-llm", action="store_true", help="template replies only")
    ap.add_argument("--tau", type=float, default=TAU_CONF,
                    help=f"abstain threshold on top-1 confidence (default {TAU_CONF}; 0 disables)")
    ap.add_argument("--no-mood", action="store_true", help="per-turn reactive baseline instead of MoodManager")
    args = ap.parse_args()

    true_labels = None
    if args.script:
        speaker, clips, true_labels = scripted_clips()
        print(f"Scripted conversation: test speaker {speaker}, labels {' '.join(LABEL_SCRIPT)}")
    elif args.clips:
        clips = []
        for c in map(Path, args.clips):   # PowerShell doesn't expand *.wav, so accept folders too
            clips += sorted(c.glob("*.wav")) if c.is_dir() else [c]
        if not clips:
            ap.error("no .wav files found")
    else:
        ap.error("give wav files or --script")

    agent = Agent(use_llm=not args.no_llm, use_mood=not args.no_mood, tau_conf=args.tau)
    results = [agent.run_turn(c) for c in clips]
    summary(results, true_labels)


if __name__ == "__main__":
    main()
