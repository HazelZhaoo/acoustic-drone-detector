"""Live drone detection from the microphone (or a recording, for testing).

Every 0.48 s it takes the latest 0.96 s of sound, runs it through the same
steps as training (16 kHz mono -> YAMNet -> classifier) and prints the drone
probability. It only says DRONE when at least SMOOTH_HITS of the last
SMOOTH_WINDOWS windows agree, so a single odd sound doesn't trigger the alarm.

Run:
  cd src && ../.venv/bin/python live.py              # microphone (Ctrl+C to stop)
  cd src && ../.venv/bin/python live.py clip.wav     # a recording instead
"""
import queue
import sys
from collections import deque
from pathlib import Path

import joblib
import numpy as np

import audio
import yamnet

ROOT = Path(__file__).parent.parent
SMOOTH_WINDOWS = 5
SMOOTH_HITS = 3


class Detector:
    def __init__(self):
        saved = joblib.load(ROOT / "models/classifier.joblib")
        self.model, self.threshold = saved["model"], saved["threshold"]
        self.recent = deque(maxlen=SMOOTH_WINDOWS)
        yamnet.model()  # load YAMNet before listening starts

    def __call__(self, window: np.ndarray, t: float) -> None:
        p = self.model.predict_proba(yamnet.embed(window[None]))[0, 1]
        self.recent.append(p >= self.threshold)
        alarm = sum(self.recent) >= SMOOTH_HITS
        bar = "█" * int(p * 20)
        status = "🚨 DRONE" if alarm else "   —"
        print(f"{t:6.1f}s  {p:5.2f} {bar:<20}  {status}", flush=True)


def run_file(path: str, detect: Detector) -> None:
    x = audio.load(path)
    hop = int(audio.HOP_S * audio.SAMPLE_RATE)
    for i, w in enumerate(audio.windows(x)):
        detect(w, i * hop / audio.SAMPLE_RATE)


def run_mic(detect: Detector) -> None:
    import sounddevice as sd

    sr = int(sd.query_devices(kind="input")["default_samplerate"])
    win = int(audio.WINDOW_S * sr)
    hop = int(audio.HOP_S * sr)
    chunks = queue.Queue()
    buffer = np.zeros(0, np.float32)
    t = 0.0
    print(f"Listening at {sr} Hz (resampled to {audio.SAMPLE_RATE}). Ctrl+C to stop.\n")
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=hop,
                        callback=lambda data, *_: chunks.put(data[:, 0].copy())):
        while True:
            buffer = np.concatenate([buffer, chunks.get()])[-win:]
            if buffer.size == win:
                detect(audio.to_16k(buffer, sr)[: int(audio.WINDOW_S * audio.SAMPLE_RATE)], t)
                t += audio.HOP_S


if __name__ == "__main__":
    detector = Detector()
    try:
        run_file(sys.argv[1], detector) if len(sys.argv) > 1 else run_mic(detector)
    except KeyboardInterrupt:
        print("\nStopped.")
