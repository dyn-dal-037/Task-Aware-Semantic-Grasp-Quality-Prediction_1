"""
Evaluation script.
Loads both checkpoints, runs test set evaluation, prints comparison table.
Run: python evaluate/evaluate.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset.dataloader import build_dataloaders
from models.baseline    import ResNetFiLM
from models.physics     import ResNetFiLMPhysics
from losses.losses      import BaselineLoss, PhysicsLoss
from train.trainer      import Trainer
from utils.metrics      import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_task_breakdown,
    compute_object_breakdown,
    grasp_quality_score
)
from utils.visualization import (
    plot_confusion_matrix,
    plot_calibration_curve,
    plot_gqs_distribution,
    plot_per_object_accuracy
)


def load_checkpoint(ckpt_path: str):
    return torch.load(ckpt_path, map_location="cpu")


def evaluate_model(model, loaders, cfg, is_physics, device):
    """Run full test set inference, collect all outputs."""
    model.eval()
    loader = loaders["test"]

    all_labels, all_probs, all_stab_t, all_stab_p = [], [], [], []
    all_physics_p = []
    all_tasks, all_objects = [], []

    with torch.no_grad():
        for batch in loader:
            image       = batch["image"].to(device)
            task_id     = batch["task_id"].to(device)
            grasp_point = batch["grasp_point"].to(device)
            label       = batch["label"].numpy()
            stability   = batch["stability"].numpy()

            if is_physics:
                logit_cls, pred_stab, pred_physics = model(image, task_id, grasp_point)
                all_physics_p.extend(torch.sigmoid(pred_physics).cpu().numpy().flatten())
            else:
                logit_cls, pred_stab = model(image, task_id, grasp_point)

            probs  = torch.sigmoid(logit_cls).cpu().numpy().flatten()
            stabp  = torch.sigmoid(pred_stab).cpu().numpy().flatten()

            all_labels.extend(label.tolist())
            all_probs.extend(probs.tolist())
            all_stab_t.extend(stability.tolist())
            all_stab_p.extend(stabp.tolist())
            all_tasks.extend(batch["task_name"])
            all_objects.extend(batch["object_name"])

    labels   = np.array(all_labels)
    probs    = np.array(all_probs)
    stab_t   = np.array(all_stab_t)
    stab_p   = np.array(all_stab_p)
    phys_p   = np.array(all_physics_p) if all_physics_p else None

    cls_m  = compute_classification_metrics(labels, probs)
    reg_m  = compute_regression_metrics(stab_t, stab_p)
    task_m = compute_task_breakdown(labels, probs, all_tasks)
    obj_m  = compute_object_breakdown(labels, probs, all_objects)

    return {
        "cls":     cls_m,
        "reg":     reg_m,
        "task":    task_m,
        "obj":     obj_m,
        "labels":  labels,
        "probs":   probs,
        "stab_t":  stab_t,
        "stab_p":  stab_p,
        "phys_p":  phys_p,
        "tasks":   all_tasks,
        "objects": all_objects,
    }


def print_comparison_table(baseline_r: dict, physics_r: dict):
    metrics = [
        ("Accuracy",    "cls",  "accuracy"),
        ("Precision",   "cls",  "precision"),
        ("Recall",      "cls",  "recall"),
        ("F1 Score",    "cls",  "f1"),
        ("AUROC",       "cls",  "auroc"),
        ("MSE (stab)",  "reg",  "mse"),
        ("MAE (stab)",  "reg",  "mae"),
        ("Spearman ρ",  "reg",  "spearman"),
    ]

    print("\n" + "=" * 65)
    print(f"  {'Metric':<25} {'Baseline':>15} {'Physics':>15}  {'Δ':>6}")
    print("=" * 65)
    for name, group, key in metrics:
        b = baseline_r[group][key]
        p = physics_r[group][key]
        delta = p - b
        arrow = "↑" if delta > 0 else "↓"
        print(f"  {name:<25} {b:>15.4f} {p:>15.4f}  {arrow}{abs(delta):.4f}")
    print("=" * 65)

    # Per-task breakdown
    print("\n── Per-Task Accuracy ──")
    all_tasks = set(list(baseline_r["task"].keys()) + list(physics_r["task"].keys()))
    for t in sorted(all_tasks):
        b = baseline_r["task"].get(t, 0)
        p = physics_r["task"].get(t, 0)
        print(f"  {t:<20} Baseline: {b:.4f}  Physics: {p:.4f}")

    # Per-object breakdown
    print("\n── Per-Object Accuracy ──")
    all_objs = set(list(baseline_r["obj"].keys()) + list(physics_r["obj"].keys()))
    for o in sorted(all_objs):
        b = baseline_r["obj"].get(o, 0)
        p = physics_r["obj"].get(o, 0)
        print(f"  {o:<20} Baseline: {b:.4f}  Physics: {p:.4f}")


def save_comparison_plots(baseline_r: dict, physics_r: dict, out_dir: str = "evaluate/"):
    os.makedirs(out_dir, exist_ok=True)
    inf_alpha = 0.60

    for model_name, r in [("Baseline", baseline_r), ("Physics", physics_r)]:
        preds = (r["probs"] >= 0.5).astype(int)

        cm_fig  = plot_confusion_matrix(r["labels"], preds, f"{model_name} — Confusion Matrix")
        cal_fig = plot_calibration_curve(r["labels"], r["probs"], f"{model_name} — Calibration")
        obj_fig = plot_per_object_accuracy(r["obj"], f"{model_name} — Per-Object Accuracy")

        gqs = inf_alpha * r["probs"] + (1 - inf_alpha) * r["stab_p"]
        gqs_fig = plot_gqs_distribution(gqs, r["labels"], f"{model_name} — GQS Distribution")

        for fig, name in [(cm_fig,  "confusion"), (cal_fig, "calibration"),
                          (obj_fig, "per_object"), (gqs_fig, "gqs_dist")]:
            fig.savefig(os.path.join(out_dir, f"{model_name.lower()}_{name}.png"), dpi=150)
            plt.close(fig)

    # Side-by-side bar chart comparison
    metrics_to_plot = ["accuracy", "f1", "auroc"]
    b_vals = [baseline_r["cls"][m] for m in metrics_to_plot]
    p_vals = [physics_r["cls"][m]  for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, b_vals, 0.4, label="Baseline", color="#4C72B0")
    ax.bar(x + 0.2, p_vals, 0.4, label="Physics",  color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_to_plot])
    ax.set_ylim(0, 1.1)
    ax.set_title("Baseline vs Physics — Classification Metrics")
    ax.legend()
    for i, (b, p) in enumerate(zip(b_vals, p_vals)):
        ax.text(i - 0.2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, p + 0.01, f"{p:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_bar.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {out_dir}")


def run_evaluation(cfg_path: str = "config/config.yaml", data_dir: str = "data/"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(cfg, data_dir)

    results = {}
    for model_type, is_physics in [("baseline", False), ("physics", True)]:
        ckpt_path = f"checkpoints/{model_type}/best_model.pt"
        if not os.path.exists(ckpt_path):
            print(f"No checkpoint for {model_type} at {ckpt_path}. Skipping.")
            continue

        ckpt = load_checkpoint(ckpt_path)
        if model_type == "baseline":
            model = ResNetFiLM(cfg)
        else:
            model = ResNetFiLMPhysics(cfg)

        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        print(f"\nEvaluating {model_type}...")
        r = evaluate_model(model, loaders, cfg, is_physics, device)
        results[model_type] = r

    if "baseline" in results and "physics" in results:
        print_comparison_table(results["baseline"], results["physics"])
        save_comparison_plots(results["baseline"], results["physics"])

    # Save numerical results
    save_path = "evaluate/results.json"
    serializable = {}
    for m, r in results.items():
        serializable[m] = {k: v for k, v in r.items()
                           if not isinstance(v, np.ndarray)}
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    run_evaluation()
