"""Audio preprocessing shared by training and the Raspberry Pi.

Both sides must turn sound into model input the exact same way, so every
step lives here and nowhere else.
"""
import io

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000   # YAMNet expects 16 kHz mono
WINDOW_S = 0.96        # one YAMNet frame
HOP_S = 0.48           # 50% overlap between windows
MIN_CLIP_S = 0.5       # shorter clips are dropped; 0.5–0.96 s clips are looped to fill a window


def load(source) -> np.ndarray:
    """Read a WAV file path or raw bytes into mono float32 at 16 kHz."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # stereo -> mono
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio


def rms(audio: np.ndarray) -> float:
    """Loudness; used to spot silent clips."""
    return float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0


def windows(audio: np.ndarray, max_windows: int | None = None) -> np.ndarray:
    """Cut audio into overlapping 0.96 s windows, shape (n, 15360).

    Clips between 0.5 s and 0.96 s are looped (not zero-padded) to fill one
    window: most drone clips are 0.5 s while non-drone clips are long, so
    silence padding would let the model learn "half-silent window = drone",
    which never happens with a live microphone.
    max_windows caps long recordings so a few long files can't dominate training.
    """
    win = int(WINDOW_S * SAMPLE_RATE)
    hop = int(HOP_S * SAMPLE_RATE)
    if audio.size < win:
        audio = np.tile(audio, int(np.ceil(win / audio.size)))[:win]
    starts = range(0, audio.size - win + 1, hop)
    out = np.stack([audio[s : s + win] for s in starts])
    if max_windows and len(out) > max_windows:
        # Spread picks across the whole clip instead of taking only the start.
        out = out[np.linspace(0, len(out) - 1, max_windows).astype(int)]
    return out
