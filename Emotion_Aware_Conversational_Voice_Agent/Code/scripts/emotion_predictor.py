"""
emotion_predictor.py
Per-clip emotion probabilities from the saved XGBoost baseline.
Reads: checkpoints/xgb_baseline.json, checkpoints/xgb_meta.json
The agent only uses predict_file / predict_array, so the model behind them can be swapped later.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from feature_extraction import load_audio, preprocess, extract_features

# Config
CODE_DIR = Path(__file__).resolve().parent.parent   # .../Code
MODEL_PATH = CODE_DIR / "checkpoints" / "xgb_baseline.json"
META_PATH = CODE_DIR / "checkpoints" / "xgb_meta.json"
DISTRESS = ["angry", "sad"]


class EmotionPredictor:
    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH):
        meta = json.loads(Path(meta_path).read_text())
        self.classes = meta["classes"]            # e.g. ["angry", "happy", "sad"]
        self.feature_cols = meta["feature_cols"]  # column order the model was trained on
        self.model = XGBClassifier()
        self.model.load_model(model_path)

    def predict_array(self, y: np.ndarray, sr: int) -> dict:
        feats = extract_features(preprocess(y, sr), sr)
        row = pd.DataFrame([feats])[self.feature_cols]   # select by name, in training order
        proba = self.model.predict_proba(row)[0]
        probs = {c: float(p) for c, p in zip(self.classes, proba)}
        return {
            "probs": probs,
            "label": self.classes[int(np.argmax(proba))],
            "p_distress": sum(probs[c] for c in DISTRESS),
        }

    def predict_file(self, path) -> dict:
        y, sr = load_audio(path)
        return self.predict_array(y, sr)
