"""Plots for the trained classifier (run after train.py).

Writes to docs/:
  confusion_matrix.png   predicted vs. actual on test
  score_distribution.png how confident the model is, per kind of window
  precision_recall.png   the trade-off between missed drones and false alarms

Run:  cd src && ../.venv/bin/python evaluate.py
"""
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
QUIET_BLOCKS = ["drone:6", "drone:7"]  # all long, quiet drone recordings (see README)

# Validated categorical palette (light surface), in fixed order.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
NOT_DRONE, DRONE, DRONE_QUIET = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def predict(split: str):
    d = np.load(ROOT / f"data/processed/{split}.npz")
    clf = joblib.load(ROOT / "models/classifier.joblib")
    return d["y"], clf["model"].predict_proba(d["X"])[:, 1], d["block"], clf["threshold"]


def plot_confusion(y, p, t):
    cm = confusion_matrix(y, (p >= t).astype(int))
    share = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    labels = ["Not drone", "Drone"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            dark = share[i, j] > 0.5
            ax.text(j, i, f"{share[i, j]:.1%}\n{cm[i, j]:,} windows", ha="center", va="center",
                    color="#ffffff" if dark else INK, fontsize=11)
    ax.set_title("Test set: predicted vs. actual\n(each row sums to 100%)", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(DOCS / "confusion_matrix.png", dpi=160)


def plot_scores(y, p, block, t):
    quiet = np.isin(block, QUIET_BLOCKS)
    groups = [
        ("Not drone", p[y == 0], NOT_DRONE),
        ("Drone (typical clips)", p[(y == 1) & ~quiet], DRONE),
        ("Drone (long, quiet recordings)", p[(y == 1) & quiet], DRONE_QUIET),
    ]
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, scores, color in groups:
        weights = np.full(len(scores), 1 / len(scores))
        ax.hist(scores, bins=bins, weights=weights, histtype="step", linewidth=2, color=color, label=name)
    ax.axvline(t, color=INK_2, linewidth=1, linestyle="--")
    ax.text(t + 0.01, ax.get_ylim()[1] * 0.95, f"alarm threshold {t:.2f}", color=INK_2, va="top", fontsize=10)
    ax.set_xlabel("Model's drone probability for a window")
    ax.set_ylabel("Share of windows in group")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.42, 0.55))
    ax.set_title("Test set: the long, quiet drone recordings mostly score low", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(DOCS / "score_distribution.png", dpi=160)


def plot_pr(y_val, p_val, y_test, p_test, t):
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for name, y, p, color, style in [("Validation", y_val, p_val, INK_2, "--"), ("Test", y_test, p_test, NOT_DRONE, "-")]:
        precision, recall, thresholds = precision_recall_curve(y, p)
        ax.plot(recall, precision, color=color, linestyle=style, linewidth=2, label=name)
        i = np.searchsorted(thresholds, t)
        ax.plot(recall[i], precision[i], "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xlabel("Recall (share of drone windows caught)")
    ax.set_ylabel("Precision (share of alarms that were drones)")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0.4, 1.01)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Precision vs. recall (dot = chosen threshold)", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(DOCS / "precision_recall.png", dpi=160)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    y_val, p_val, _, t = predict("val")
    y_test, p_test, block_test, _ = predict("test")
    plot_confusion(y_test, p_test, t)
    plot_scores(y_test, p_test, block_test, t)
    plot_pr(y_val, p_val, y_test, p_test, t)
    print("Wrote", *sorted(p.name for p in DOCS.glob("*.png")))
