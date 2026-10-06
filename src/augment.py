"""Step 2b: simulate quiet, far-away drones and embed them for training.

The first model missed most quiet, long-range drone recordings (see README):
it had only heard loud, close drones. This makes an altered copy of every
training drone clip, the way distance changes a real drone's sound:
  1. muffle it (low-pass filter: air absorbs high pitches first),
  2. mix in a random non-drone training clip at a random signal-to-noise ratio,
  3. turn the result down to a random volume.
Some non-drone clips also get random volume changes, so loudness alone can't
tell the classes apart.

Only the train split is used, for both drones and background noise, so
validation and test stay untouched.
Output: data/processed/train_aug.npz (same fields as train.npz).

Run:  cd src && ../.venv/bin/python augment.py      (~30 min on a laptop CPU, resumable)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import butter, sosfilt

import audio
import yamnet

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data/raw/data"
MANIFEST = ROOT / "data/processed/manifest.parquet"
PARTS = ROOT / "data/processed/aug"
OUT = ROOT / "data/processed/train_aug.npz"

SEED = 7
NOISE_POOL = 400               # non-drone train clips used as background noise
LOWPASS_HZ = (800, 4000)       # muffling strength (lower = farther away)
LOWPASS_PROB = 0.8
SNR_DB = (-5, 20)              # drone vs. background loudness (lower = drone buried deeper)
NOISE_PROB = 0.8
GAIN_DB = (-35, -5)            # final volume change
NON_DRONE_SHARE = 0.3          # share of non-drone train clips that get a volume-only copy
MAX_WINDOWS = 30


def load_clip(shard: str, row: int, cache: dict) -> np.ndarray:
    if shard not in cache:
        cache.clear()  # keep one shard in memory at a time
        cache[shard] = pq.read_table(RAW / shard, columns=["audio"]).column("audio")
    return audio.load(cache[shard][row].as_py()["bytes"])


def db_to_amp(db: float) -> float:
    return 10 ** (db / 20)


def far_away(x: np.ndarray, noise_pool: list, rng: np.random.Generator) -> np.ndarray:
    """Make a drone clip sound quieter, muffled and buried in background noise."""
    if rng.random() < LOWPASS_PROB:
        sos = butter(4, rng.uniform(*LOWPASS_HZ), btype="low", fs=audio.SAMPLE_RATE, output="sos")
        x = sosfilt(sos, x).astype(np.float32)
    if rng.random() < NOISE_PROB:
        noise = noise_pool[rng.integers(len(noise_pool))]
        start = rng.integers(max(1, noise.size - x.size))
        noise = np.resize(noise[start:], x.size)  # repeat if the noise clip is shorter
        snr = rng.uniform(*SNR_DB)
        scale = audio.rms(x) / max(audio.rms(noise), 1e-6) / db_to_amp(snr)
        x = x + scale * noise
    x = x * db_to_amp(rng.uniform(*GAIN_DB))
    return np.clip(x, -1, 1).astype(np.float32)


def augment_shard(shard: str, clips: pd.DataFrame, noise_pool: list, rng: np.random.Generator) -> None:
    out = PARTS / f"{Path(shard).stem}.npz"
    if out.exists():
        return
    cache = {}
    X, y, block, clip = [], [], [], []
    for idx, c in clips.iterrows():
        x = load_clip(shard, c.row, cache)
        if c.label == 1:
            x = far_away(x, noise_pool, rng)
        else:
            x = (x * db_to_amp(rng.uniform(*GAIN_DB))).astype(np.float32)
        wins = audio.windows(x, MAX_WINDOWS)
        X.append(yamnet.embed(wins))
        y += [c.label] * len(wins)
        block += [c.block] * len(wins)
        clip += [idx] * len(wins)
    np.savez(out, X=np.concatenate(X), y=np.array(y, np.int8), block=np.array(block), clip=np.array(clip))


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    PARTS.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(MANIFEST).query("keep and split == 'train'")
    non_drone = train[train.label == 0]

    # Background noise comes from non-drone *train* clips only.
    cache = {}
    picks = non_drone.sample(NOISE_POOL, random_state=SEED).sort_values("shard")
    noise_pool = [load_clip(c.shard, c.row, cache) for c in picks.itertuples()]

    selected = pd.concat([train[train.label == 1],
                          non_drone.sample(frac=NON_DRONE_SHARE, random_state=SEED)])
    start = time.time()
    shards = sorted(selected.shard.unique())
    for i, (shard, clips) in enumerate(selected.groupby("shard"), 1):
        augment_shard(shard, clips, noise_pool, np.random.default_rng([SEED, i]))
        print(f"[{i}/{len(shards)}] {shard} done  ({(time.time() - start) / 60:.1f} min)", flush=True)

    parts = [np.load(p) for p in sorted(PARTS.glob("*.npz"))]
    np.savez(OUT, **{k: np.concatenate([p[k] for p in parts]) for k in ["X", "y", "block", "clip"]})
    d = np.load(OUT)
    print(f"Wrote {OUT}: {len(d['y'])} windows, {int(d['y'].sum())} drone")
