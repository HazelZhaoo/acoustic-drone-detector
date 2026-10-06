"""Blind test on recordings from sources the model has never seen.

  - Outdoor drones (Zenodo 15190811, CC BY 4.0): DJI Matrice 300 (large, ~9 kg)
    and DJI Mavic Mini 2 (small), at 5/8/10 m distance and 0/5/8 m height, with
    and without people talking nearby ("sekwencja" files).
  - Drone / helicopter / background clips (Zenodo 5500576, DroneDetectionThesis).

Each recording goes through the same path as the live detector (windows ->
YAMNet -> classifier -> 3-of-5 alarm). Per group it reports how often the alarm
fires at some point: wanted for drones, a false alarm for everything else.
Writes docs/blind_test.csv.

Run:  cd src && ../.venv/bin/python blind_test.py
"""
import re
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import audio
import yamnet

ROOT = Path(__file__).parent.parent
EXT = ROOT / "data/external"
MODELS = {"original": "classifier", "augmented": "classifier_aug"}


def recordings() -> pd.DataFrame:
    rows = []
    for f in sorted(EXT.glob("X4_*/*/OLYMPUSLS11/*.WAV")):
        m = re.search(r"x4_d\d+_(\w+?)_(\d+)m_(\d+)m_", f.name, re.I)
        drone = "Matrice 300 (large)" if "matrice" in m.group(1).lower() else "Mavic Mini 2 (small)"
        speech = "sekwencja" in f.name.lower()
        rows.append({"path": f, "group": f"Drone: {drone}" + (", people talking" if speech else ""),
                     "is_drone": True, "distance_m": int(m.group(2)), "height_m": int(m.group(3))})
    for f in sorted(EXT.glob("thesis/*/Data/Audio/*.wav")):
        kind = f.name.split("_")[0].lower()
        rows.append({"path": f, "group": {"drone": "Drone: thesis set", "helicopter": "Helicopter",
                                          "background": "Background"}[kind],
                     "is_drone": kind == "drone", "distance_m": None, "height_m": None})
    return pd.DataFrame(rows)


def alarm_fired(hits: np.ndarray) -> bool:
    recent = deque(maxlen=5)
    for h in hits:
        recent.append(h)
        if sum(recent) >= 3:
            return True
    return False


if __name__ == "__main__":
    recs = recordings()
    clfs = {name: joblib.load(ROOT / f"models/{file}.joblib") for name, file in MODELS.items()}
    results = []
    for i, r in enumerate(recs.itertuples(), 1):
        emb = yamnet.embed(audio.windows(audio.load(r.path)))
        row = r._asdict()
        for name, c in clfs.items():
            hits = c["model"].predict_proba(emb)[:, 1] >= c["threshold"]
            row[f"{name}_alarm"] = alarm_fired(hits)
            row[f"{name}_window_share"] = float(hits.mean())
        results.append(row)
        if i % 30 == 0:
            print(f"{i}/{len(recs)} recordings", flush=True)

    df = pd.DataFrame(results).drop(columns=["Index"])
    df["path"] = df.path.map(lambda p: Path(p).name)
    (ROOT / "docs").mkdir(exist_ok=True)
    df.to_csv(ROOT / "docs/blind_test.csv", index=False)

    summary = df.groupby("group").agg(recordings=("path", "size"), is_drone=("is_drone", "first"),
                                      original=("original_alarm", "mean"), augmented=("augmented_alarm", "mean"))
    print("\nShare of recordings where the alarm fired (drones: want high; others: false alarms, want low)")
    print(summary.sort_values("is_drone", ascending=False).to_string(formatters={
        "original": "{:.0%}".format, "augmented": "{:.0%}".format}))
    outdoor = df[df.distance_m.notna()]
    print("\nOutdoor drones by distance (window share flagged, original model):")
    print(outdoor.groupby(["group", "distance_m"]).original_window_share.mean().unstack().round(2).to_string())
