from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
def plot_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    label: str = "model",
    title: str = "Precision–Recall curve (validation set)",
    save_path: str | None = None,
    highlight_thresholds: list[float] | None = None,
) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})", linewidth=2)

    if highlight_thresholds:
        for thr in highlight_thresholds:
            idx = np.searchsorted(thresholds, thr)
            idx = min(idx, len(precision) - 2)
            ax.scatter(
                recall[idx], precision[idx],
                zorder=5, s=80,
                label=f"thr={thr:.2f}  P={precision[idx]:.2f} R={recall[idx]:.2f}",
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Average Precision (AP): {ap:.4f}")

def evaluate_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: list[float],
    label_names: list[str] | None = None,
    pos_label: int = 1,
) -> list[dict]:
    results = []
    label_names = label_names or ["class_0", "class_1"]

    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        results.append({
            "threshold": thr,
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0), 4),
            "macro_f1":  round(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0), 4),
            "conf_matrix": cm,
            "y_pred":    y_pred,
        })

    return results


def print_threshold_table(results: list[dict]) -> None:
    header = f"{'Threshold':>10}  {'Accuracy':>8}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'MacroF1':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['threshold']:>10.3f}  "
            f"{r['accuracy']:>8.4f}  "
            f"{r['precision']:>9.4f}  "
            f"{r['recall']:>6.4f}  "
            f"{r['f1']:>6.4f}  "
            f"{r['macro_f1']:>7.4f}"
        )


def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    strategy: str = "f1",
    pos_label: int = 1,
    n_candidates: int = 200,
) -> tuple[float, dict]:
    lo, hi = float(scores.min()), float(scores.max())
    candidates = np.linspace(lo, hi, n_candidates)
    evaluated = evaluate_thresholds(y_true, scores, candidates.tolist(), pos_label=pos_label)

    key_map = {
        "f1":       "f1",
        "precision": "precision",
        "recall":   "recall",
        "macro_f1": "macro_f1",
    }
    key = key_map.get(strategy, "f1")
    best = max(evaluated, key=lambda r: r[key])
    return best["threshold"], best
