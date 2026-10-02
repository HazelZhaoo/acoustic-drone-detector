# Drone Detector

An offline acoustic early-warning device: a microphone listens, a small model decides "drone / not drone", and a local alarm gives people time to take cover. No internet needed.

**Status:** M1 in progress. Data prep, YAMNet embeddings and a first classifier are done (results below); next is a live laptop-mic demo.

## Plan

| Milestone | Done when |
|---|---|
| **M1** It hears a drone | Laptop mic says "DRONE" when a drone video plays |
| M2 The honest test | Accuracy on the dataset *and* on our own everyday-noise recordings |
| M3 It lives in a box | Raspberry Pi + mic + buzzer, no laptop |
| M4 It gets smarter | Retrained on noise recorded through the Pi's mic; fewer false alarms |
| M5 Village warns village | Two LoRa (Meshtastic) nodes: one hears, the other alarms |

## Training flow

```
data/raw/            39 parquet shards (Hugging Face: geronimobasso/drone-audio-detection-samples)
   │  src/prepare.py   decode → clean → split by blocks
   ▼
data/processed/manifest.parquet   one row per clip: stats, keep/drop reason, split
   │  src/embed.py     clip → 0.96 s windows → YAMNet → 1024 numbers per window
   ▼
data/processed/{train,val,test}.npz
   │  src/train.py     logistic regression → tune threshold on val → test once
   ▼
models/classifier
```

`src/audio.py` (16 kHz mono, 0.96 s windows, 0.48 s hop) and `src/yamnet.py` (embeddings) are shared by training and the Pi, so both see sound the same way. YAMNet runs one window per call on purpose: batching windows into one call is ~6x faster but changes the embeddings slightly (cosine similarity 0.93–0.99), which would no longer match the live device.

### Cleaning (prepare.py)
- **Unreadable** files are dropped.
- **Too short** (< 0.5 s) clips are dropped.
- **Silent "drone" clips** are dropped: silence labeled drone would teach the wrong thing.
- **Duplicates** (same decoded audio under a different name) are dropped.
- Every dropped clip keeps its reason in the manifest.

### Splitting (prepare.py)
The dataset renamed files to sequential numbers, so clips cut from one recording sit next to each other. We split **blocks of 200 consecutive clips** (per class) into train/val/test 70/15/15, so one recording can't be in both train and test and inflate the score.

### Known risks to watch
- **Shortcut learning.** Drone clips are mostly 0.5 s from a few drone datasets; non-drone clips are long city/nature recordings. The model may learn "which dataset" instead of "is there a drone". Short clips are looped (not zero-padded) to avoid one obvious shortcut; M2's own-recording test is the real check.
- **Domain shift.** Public drone audio is clean and close-range; real conditions aren't. See *Acoustic UAV Detection in Battlefield Scenarios* (arXiv 2608.14287), where baselines dropped to ~55% F1 on real recordings.
- **Class balance.** By clips it's 10:1 drone, but by 0.96 s windows it's roughly balanced (train: ~163k not-drone vs ~114k drone), because non-drone clips are long. Still use class weights and judge by precision/recall, not accuracy.


## Results so far (M1)

Logistic regression on YAMNet embeddings, one prediction per 0.96 s window. C and the alarm threshold (0.71) were chosen on validation only; test was scored once.

| Split | Precision | Recall | F1 | False-alarm rate |
|---|---|---|---|---|
| Validation | 99.4% | 99.6% | 99.5% | 0.6% |
| **Test** | **98.5%** | **78.7%** | **87.5%** | **1.0%** |
| Test without the two long-recording groups | 98.4% | 99.4% | – | – |

**What the gap means.** Almost all missed drones come from two recording groups (`drone:6`, `drone:7`). They hold *every* long drone recording in the dataset (258 clips, up to 5 min) and are much quieter than the rest (median RMS 0.03–0.04 vs ~0.2), likely a different source such as real flights recorded at a distance. The block split put both groups in test, so the model never trained on anything like them and misses 87% of their windows.

So the model has learned "loud, close-range drone clips" very well and does not yet generalize to quieter, longer, real-flight-like recordings, which is exactly the condition a deployed sensor faces. This is the domain-shift problem described in *Acoustic UAV Detection in Battlefield Scenarios* (arXiv 2608.14287), showing up in public data.

**Next steps to address it**
- Volume (gain) augmentation and mixing drone clips with background noise at different levels, so loudness stops being a shortcut.
- A source-aware evaluation: hold out whole recording sources, not just blocks, and report per-source results.
- PCEN or similar loudness normalization before the classifier.
- Real recordings through the device microphone (M2/M4).

## Setup

```bash
~/.pyenv/versions/3.12.0/bin/python -m venv .venv
.venv/bin/pip install huggingface_hub pyarrow pandas numpy soundfile scipy scikit-learn \
    tensorflow tensorflow-hub "setuptools<81"   # tensorflow-hub still imports pkg_resources
cd src
../.venv/bin/python prepare.py   # minutes
../.venv/bin/python embed.py     # ~1 hour on a laptop CPU; resumable
../.venv/bin/python train.py
```

## Data
[drone-audio-detection-samples](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples): ~180k clips, ~61 h (27 h drone, 34 h not drone), 16 kHz mono. Combines Al-Emadi's DroneAudioDataset, DREGON, SPCup19 and others (drone) with UrbanSound8K, TUT 2017, ESC-50 and others (not drone). Check each source's license before any commercial use.
