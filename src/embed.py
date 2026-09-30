"""Step 2: turn every kept clip into YAMNet embeddings.

Reads data/processed/manifest.parquet (from prepare.py) and, for each kept
clip, loads its audio, cuts it into 0.96 s windows (at most MAX_WINDOWS per
clip) and embeds each window with YAMNet.

Work is saved per raw shard in data/processed/emb/, so an interrupted run
picks up where it stopped. At the end the shards are combined into
data/processed/{train,val,test}.npz with:
  X      (n, 1024) embeddings, one row per window
  y      label per window (1 = drone, 0 = not drone)
  block  recording group per window (for per-recording error analysis)
  clip   manifest row index per window (to trace a window back to its clip)

Run:  cd src && ../.venv/bin/python embed.py      (about an hour on a laptop CPU)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import audio
import yamnet

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data/raw/data"
MANIFEST = ROOT / "data/processed/manifest.parquet"
PARTS = ROOT / "data/processed/emb"
MAX_WINDOWS = 30  # cap per clip so a few long recordings can't dominate


def embed_shard(shard: str, clips: pd.DataFrame) -> None:
    out = PARTS / f"{Path(shard).stem}.npz"
    if out.exists():
        return  # done in an earlier run
    audio_col = pq.read_table(RAW / shard, columns=["audio"]).column("audio")
    X, y, block, clip, split = [], [], [], [], []
    for idx, c in clips.iterrows():
        wins = audio.windows(audio.load(audio_col[c.row].as_py()["bytes"]), MAX_WINDOWS)
        X.append(yamnet.embed(wins))
        n = len(wins)
        y += [c.label] * n
        block += [c.block] * n
        clip += [idx] * n
        split += [c.split] * n
    np.savez(out, X=np.concatenate(X), y=np.array(y, np.int8), block=np.array(block),
             clip=np.array(clip), split=np.array(split))


def combine() -> None:
    parts = [np.load(p) for p in sorted(PARTS.glob("*.npz"))]
    fields = {k: np.concatenate([p[k] for p in parts]) for k in ["X", "y", "block", "clip", "split"]}
    for name in ["train", "val", "test"]:
        m = fields["split"] == name
        np.savez(ROOT / f"data/processed/{name}.npz",
                 X=fields["X"][m], y=fields["y"][m], block=fields["block"][m], clip=fields["clip"][m])
        print(f"{name}: {m.sum()} windows, {int(fields['y'][m].sum())} drone")


if __name__ == "__main__":
    PARTS.mkdir(parents=True, exist_ok=True)
    kept = pd.read_parquet(MANIFEST).query("keep")
    shards = sorted(kept.shard.unique())
    start = time.time()
    for i, (shard, clips) in enumerate(kept.groupby("shard"), 1):
        embed_shard(shard, clips)
        mins = (time.time() - start) / 60
        print(f"[{i}/{len(shards)}] {shard} done  ({mins:.1f} min elapsed)", flush=True)
    combine()
