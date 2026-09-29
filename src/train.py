"""Step 3 (next session): train and evaluate the drone / not-drone classifier.

Plan:
  1. Load train/val npz embeddings.
  2. Baseline: logistic regression with class_weight="balanced"
     (drone windows outnumber non-drone ones).
  3. Tune on val only: regularization, and the decision threshold
     (a false alarm costs trust, a miss costs safety; pick deliberately).
  4. Report once on test: precision, recall, F1, confusion matrix.
  5. Save the classifier to models/ for the live laptop demo and the Pi.
"""
