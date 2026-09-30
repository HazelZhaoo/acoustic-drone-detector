"""YAMNet embeddings, shared by training and the Raspberry Pi.

YAMNet is Google's pretrained sound model (AudioSet, 521 sound classes). We
keep it frozen and use its second-to-last layer: 1024 numbers describing a
0.96 s window. Our own small classifier learns drone / not drone from those.
"""
import numpy as np
import tensorflow_hub as hub

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
EMBEDDING_DIM = 1024

_model = None


def model():
    """Load YAMNet once (downloaded and cached on first use)."""
    global _model
    if _model is None:
        _model = hub.load(YAMNET_URL)
    return _model


def embed(windows: np.ndarray) -> np.ndarray:
    """(n, 15360) windows from audio.windows() -> (n, 1024) embeddings.

    Each window goes through YAMNet on its own, exactly as the live device
    will do. Gluing windows into one call is ~6x faster but gives slightly
    different embeddings (YAMNet's frames spill into the neighbouring window),
    so training would no longer match what the device sees.
    """
    yam = model()
    return np.stack([yam(w)[1].numpy()[0] for w in windows]).astype(np.float32)
