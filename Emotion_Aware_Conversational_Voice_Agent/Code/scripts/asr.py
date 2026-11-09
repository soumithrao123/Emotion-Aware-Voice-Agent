"""
asr.py
Speech-to-text with openai-whisper (tiny.en, CPU).
Audio is passed as a numpy array, so no ffmpeg is needed.
Model weights are cached by whisper in ~/.cache/whisper (outside the repo).
"""
import numpy as np
import librosa
import whisper

from feature_extraction import load_audio

# Config
MODEL_NAME = "tiny.en"
WHISPER_SR = 16000   # whisper expects 16 kHz float32 mono


class ASR:
    def __init__(self, model_name=MODEL_NAME):
        self.model = whisper.load_model(model_name, device="cpu")

    def transcribe_array(self, y: np.ndarray, sr: int) -> str:
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = librosa.to_mono(y)
        if sr != WHISPER_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=WHISPER_SR)
        result = self.model.transcribe(y, language="en", fp16=False)
        return result["text"].strip()

    def transcribe_file(self, path) -> str:
        y, sr = load_audio(path)   # untrimmed, 16 kHz mono
        return self.transcribe_array(y, sr)
