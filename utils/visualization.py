"""
Visualization utilities: confusion matrix, GQS distribution, calibration curves.
All outputs are matplotlib figures that can be logged to TensorBoard.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works on Colab/server
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve


def plot_confusion_matrix(labels: np.ndarray,
                           preds: np.ndarray,
                           title: str = "Confusion Matrix") -> plt.Figure:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Reject", "Accept"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_gqs_distribution(gqs_scores: np.ndarray,
                           labels: np.ndarray,
                           title: str = "GQS Distribution") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    pos_scores = gqs_scores[labels == 1]
    neg_scores = gqs_scores[labels == 0]
    ax.hist(pos_scores * 100, bins=30, alpha=0.6, color="green", label="Positive (label=1)")
    ax.hist(neg_scores * 100, bins=30, alpha=0.6, color="red",   label="Negative (label=0)")
    ax.axvline(80, ls="--", color="green",  lw=1.5, label="Excellent threshold")
    ax.axvline(60, ls="--", color="orange", lw=1.5, label="Acceptable threshold")
    ax.axvline(40, ls="--", color="red",    lw=1.5, label="Poor threshold")
    ax.set_xlabel("Grasp Quality Score (%)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_calibration_curve(labels: np.ndarray,
                            probs: np.ndarray,
                            title: str = "Calibration Curve") -> plt.Figure:
    fraction_pos, mean_pred = calibration_curve(labels, probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "b-o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_loss_curves(history: dict, model_name: str) -> plt.Figure:
    """Plot train vs val loss from logger history dict."""
    train_key = "train/L_total"
    val_key   = "val/L_total"

    fig, ax = plt.subplots(figsize=(8, 4))
    if train_key in history:
        steps, vals = zip(*history[train_key])
        ax.plot(steps, vals, label="Train loss")
    if val_key in history:
        steps, vals = zip(*history[val_key])
        ax.plot(steps, vals, label="Val loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} — Loss Curves")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_per_object_accuracy(breakdown: dict, title: str = "Per-Object Accuracy") -> plt.Figure:
    keys   = [k.replace("acc_", "") for k in breakdown if k.startswith("acc_")]
    values = [breakdown[f"acc_{k}"] * 100 for k in keys]
    colors = cm.tab10(np.linspace(0, 1, len(keys)))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(keys, values, color=colors)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    return fig


def fig_to_tensorboard(fig: plt.Figure):
    """Convert matplotlib figure to format suitable for TB add_figure."""
    return fig
