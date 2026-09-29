"""Step 2 (next session): turn kept clips into YAMNet embeddings.

Plan:
  1. Read data/processed/manifest.parquet, keep rows where keep == True.
  2. For each clip: load(bytes) -> windows(audio, max_windows=30).
  3. Run every window through YAMNet (pretrained, frozen) -> 1024 numbers each.
  4. Save per split: data/processed/{split}.npz with X (n, 1024), y, and block
     (block lets us check results per recording later).

Needs: pip install tensorflow tensorflow-hub
"""
