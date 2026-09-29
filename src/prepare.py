"""Step 1: clean the dataset and split it into train / val / test.

Reads the downloaded parquet shards and writes data/processed/manifest.parquet:
one row per clip with its stats, whether it's kept (and why not), and its split.
Audio stays in the shards; later steps look clips up by (shard, row).

Run:  .venv/bin/python src/prepare.py
"""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from audio import MIN_CLIP_S, SAMPLE_RATE, load, rms

RAW = Path(__file__).parent.parent / "data/raw/data"
OUT = Path(__file__).parent.parent / "data/processed/manifest.parquet"

SILENCE_RMS = 1e-4   # below this a clip is effectively silent
BLOCK_SIZE = 200     # consecutive clip numbers treated as one recording group
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def scan() -> pd.DataFrame:
    """Decode every clip once and record the facts cleaning needs."""
    records = []
    for shard in sorted(RAW.glob("*.parquet")):
        table = pq.read_table(shard)
        for row, (audio, label) in enumerate(
            zip(table.column("audio").to_pylist(), table.column("label").to_pylist())
        ):
            rec = {"shard": shard.name, "row": row, "label": label, "path": audio["path"]}
            try:
                x = load(audio["bytes"])
                rec |= {
                    "duration_s": x.size / SAMPLE_RATE,
                    "rms": rms(x),
                    # Hash of the decoded audio catches duplicate clips under different names.
                    "sha1": hashlib.sha1(np.round(x, 4).tobytes()).hexdigest(),
                    "error": None,
                }
            except Exception as err:  # corrupt or unreadable file
                rec |= {"duration_s": np.nan, "rms": np.nan, "sha1": None, "error": str(err)[:80]}
            records.append(rec)
        print(f"scanned {shard.name}: {len(records)} clips so far")
    return pd.DataFrame(records)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Mark each clip kept or dropped, with the reason."""
    df["drop_reason"] = None
    df.loc[df.error.notna(), "drop_reason"] = "unreadable"
    df.loc[df.drop_reason.isna() & (df.duration_s < MIN_CLIP_S), "drop_reason"] = "too_short"
    # A silent clip labeled "drone" would teach the model that silence = drone.
    df.loc[df.drop_reason.isna() & (df.label == 1) & (df.rms < SILENCE_RMS), "drop_reason"] = "silent_drone"
    dup = df.drop_reason.isna() & df.duplicated("sha1", keep="first")
    df.loc[dup, "drop_reason"] = "duplicate"
    df["keep"] = df.drop_reason.isna()
    return df


def split(df: pd.DataFrame) -> pd.DataFrame:
    """Assign train/val/test by blocks of consecutive clip numbers, per class.

    The dataset renamed files to sequential numbers (drone-20941.wav, ...), so
    clips cut from one recording sit next to each other. Splitting whole blocks
    keeps one recording from landing in both train and test, which would make
    test accuracy look better than it really is.
    """
    df["prefix"] = df.path.str.extract(r"^(.*?)[-_]?\d+\.\w+$", expand=False)
    df["num"] = df.path.str.extract(r"(\d+)\.\w+$", expand=False).astype(int)
    df["block"] = df.prefix + ":" + (df.num // BLOCK_SIZE).astype(str)

    rng = np.random.default_rng(SEED)
    df["split"] = None
    for _, blocks in df[df.keep].groupby("label").block:
        ids = rng.permutation(blocks.unique())
        n_train = int(len(ids) * SPLIT["train"])
        n_val = int(len(ids) * SPLIT["val"])
        assign = dict.fromkeys(ids[:n_train], "train")
        assign |= dict.fromkeys(ids[n_train : n_train + n_val], "val")
        assign |= dict.fromkeys(ids[n_train + n_val :], "test")
        df.loc[blocks.index, "split"] = blocks.map(assign)
    return df


def report(df: pd.DataFrame) -> None:
    print("\nDropped clips by reason:")
    print(df.drop_reason.value_counts(dropna=True).to_string())
    kept = df[df.keep]
    print("\nKept clips and hours per split and label:")
    summary = kept.groupby(["split", "label"]).agg(clips=("row", "size"), hours=("duration_s", "sum"))
    summary["hours"] = (summary.hours / 3600).round(2)
    print(summary.to_string())


if __name__ == "__main__":
    manifest = split(clean(scan()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(OUT, index=False)
    report(manifest)
    print(f"\nWrote {OUT}")
