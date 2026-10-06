"""Step 3: train and evaluate the drone / not-drone classifier.

  1. Load the train / val / test embeddings from embed.py.
  2. Model: standardize the 1024 numbers, then logistic regression with
     class_weight="balanced".
  3. Pick the regularization strength C and the alarm threshold on val only.
     The threshold is the probability above which a window counts as "drone":
     higher = fewer false alarms but more missed drones.
  4. Score once on test and save the model + metrics to models/.

Run:  cd src && ../.venv/bin/python train.py          # original training data
      cd src && ../.venv/bin/python train.py --aug    # + simulated far-away drones (augment.py)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/processed"
MODELS = ROOT / "models"
C_GRID = [0.01, 0.1, 1.0]
QUIET_BLOCKS = ["drone:6", "drone:7"]  # the long, quiet test recordings the first model missed


def load(split: str):
    d = np.load(DATA / f"{split}.npz")
    return d["X"], d["y"], d["block"]


def fit(X, y, C: float):
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
    return model.fit(X, y)


def best_threshold(y, p) -> float:
    """Threshold with the best F1 on val (balances missed drones vs false alarms)."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


def scores(y, p, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "precision": round(precision_score(y, pred), 4),  # of alarms, how many were drones
        "recall": round(recall_score(y, pred), 4),        # of drones, how many we caught
        "f1": round(f1_score(y, pred), 4),
        "false_alarm_rate": round(fp / (fp + tn), 4),     # share of non-drone windows flagged
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def worst_blocks(y, p, block, threshold: float, n: int = 5) -> list:
    """Recording groups with the most errors: where to look first."""
    wrong = (p >= threshold).astype(int) != y
    rows = []
    for b in np.unique(block):
        m = block == b
        rows.append((b, int(m.sum()), round(float(wrong[m].mean()), 3)))
    return sorted(rows, key=lambda r: -r[2])[:n]


if __name__ == "__main__":
    use_aug = "--aug" in sys.argv
    X_train, y_train, _ = load("train")
    if use_aug:
        X_aug, y_aug, _ = load("train_aug")
        X_train, y_train = np.concatenate([X_train, X_aug]), np.concatenate([y_train, y_aug])
        print(f"Training with augmented data: {len(y_train)} windows")
    X_val, y_val, _ = load("val")

    print("Tuning C on val (PR-AUC):")
    best = None
    for C in C_GRID:
        m = fit(X_train, y_train, C)
        ap = average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        print(f"  C={C}: {ap:.4f}")
        if best is None or ap > best[1]:
            best = (C, ap, m)
    C, _, model = best

    p_val = model.predict_proba(X_val)[:, 1]
    threshold = best_threshold(y_val, p_val)
    print(f"\nChosen C={C}, threshold={threshold:.3f}")
    print("Val:", scores(y_val, p_val, threshold))

    X_test, y_test, block_test = load("test")
    p_test = model.predict_proba(X_test)[:, 1]
    test = scores(y_test, p_test, threshold)
    quiet = np.isin(block_test, QUIET_BLOCKS) & (y_test == 1)
    test["quiet_recall"] = round(float((p_test[quiet] >= threshold).mean()), 4)
    print("Test:", test)
    print("Worst test blocks (block, windows, error rate):")
    for row in worst_blocks(y_test, p_test, block_test, threshold):
        print("  ", row)

    MODELS.mkdir(exist_ok=True)
    name = "classifier_aug" if use_aug else "classifier"
    joblib.dump({"model": model, "threshold": threshold}, MODELS / f"{name}.joblib")
    (MODELS / f"metrics{'_aug' if use_aug else ''}.json").write_text(json.dumps(
        {"C": C, "threshold": threshold, "val": scores(y_val, p_val, threshold), "test": test}, indent=2))
    print(f"\nSaved {MODELS / (name + '.joblib')}")
