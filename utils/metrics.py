"""
Evaluation metrics for grasp quality prediction.
"""

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)


def compute_classification_metrics(labels: np.ndarray,
                                   probs: np.ndarray,
                                   threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy":          accuracy_score(labels, preds),
        "precision":         precision_score(labels, preds, zero_division=0),
        "recall":            recall_score(labels, preds, zero_division=0),
        "f1":                f1_score(labels, preds, zero_division=0),
        "auroc":             roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
        "avg_precision":     average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
    }
    return metrics


def compute_regression_metrics(targets: np.ndarray,
                                preds: np.ndarray) -> dict:
    mse  = np.mean((targets - preds) ** 2)
    mae  = np.mean(np.abs(targets - preds))
    rmse = np.sqrt(mse)

    # Rank correlation
    spear, _ = spearmanr(targets, preds)
    pear,  _ = pearsonr(targets, preds)

    return {
        "mse":      mse,
        "mae":      mae,
        "rmse":     rmse,
        "spearman": spear,
        "pearson":  pear,
    }


def compute_task_breakdown(labels: np.ndarray,
                           probs: np.ndarray,
                           task_names: list,
                           threshold: float = 0.5) -> dict:
    """Per-task accuracy breakdown."""
    unique_tasks = list(set(task_names))
    breakdown = {}
    for task in unique_tasks:
        mask = np.array([t == task for t in task_names])
        if mask.sum() == 0:
            continue
        preds = (probs[mask] >= threshold).astype(int)
        breakdown[f"acc_{task}"] = accuracy_score(labels[mask], preds)
    return breakdown


def compute_object_breakdown(labels: np.ndarray,
                              probs: np.ndarray,
                              object_names: list,
                              threshold: float = 0.5) -> dict:
    """Per-object accuracy breakdown."""
    unique_objs = list(set(object_names))
    breakdown = {}
    for obj in unique_objs:
        mask = np.array([o == obj for o in object_names])
        if mask.sum() == 0:
            continue
        preds = (probs[mask] >= threshold).astype(int)
        breakdown[f"acc_{obj}"] = accuracy_score(labels[mask], preds)
    return breakdown


def grasp_quality_score(p_cls: float, p_stab: float,
                         p_physics: float = None,
                         alpha: float = 0.60,
                         beta: float = 0.25,
                         gamma: float = 0.15) -> dict:
    """
    Fused Grasp Quality Score (GQS).
    Baseline: α·p_cls + (1-α)·p_stab
    Physics:  α·p_cls + β·p_stab + γ·p_physics
    """
    if p_physics is not None:
        gqs = alpha * p_cls + beta * p_stab + gamma * p_physics
    else:
        gqs = alpha * p_cls + (1 - alpha) * p_stab

    gqs_pct = round(float(gqs) * 100, 1)

    if gqs_pct >= 80:
        verdict = "Excellent"
    elif gqs_pct >= 60:
        verdict = "Acceptable"
    elif gqs_pct >= 40:
        verdict = "Poor"
    else:
        verdict = "Reject"

    return {
        "semantic_score":  round(float(p_cls) * 100, 1),
        "stability_score": round(float(p_stab) * 100, 1),
        "physics_score":   round(float(p_physics) * 100, 1) if p_physics else None,
        "grasp_quality":   gqs_pct,
        "verdict":         verdict
    }
