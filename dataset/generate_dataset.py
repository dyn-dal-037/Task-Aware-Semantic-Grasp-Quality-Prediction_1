"""
Synthetic dataset generator for Task-Aware Semantic Grasp Quality Prediction.
Generates balanced, leak-free datapoints with corrected stability scoring.
"""

import json
import random
import math
import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Stability Computation (NO task leakage)
# ─────────────────────────────────────────────────────────────────────────────

def geometric_score(x: int, y: int, img_size: int = 224) -> float:
    """Pure geometry — task-independent."""
    cx = cy = img_size // 2
    max_dist = math.sqrt(2) * cx

    dist_center = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    norm_center = 1.0 - dist_center / max_dist

    max_edge = img_size // 2
    dist_edge = min(x, y, img_size - x, img_size - y)
    norm_edge = min(dist_edge / max_edge, 1.0)

    return 0.6 * norm_center + 0.4 * norm_edge


def compute_stability(x: int, y: int, region: str,
                      region_priors: dict, img_size: int = 224) -> float:
    """
    Stability = 0.5 * geometric + 0.5 * region_prior
    No task modifier — prevents label leakage into regression target.
    """
    geo   = geometric_score(x, y, img_size)
    prior = region_priors.get(region, 0.60)
    score = 0.5 * geo + 0.5 * prior
    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

def in_bbox(x: int, y: int, bbox: list) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def sample_point_in_bbox(bbox: list) -> tuple:
    x1, y1, x2, y2 = bbox
    return random.randint(x1, x2), random.randint(y1, y2)


def jitter_point(x: int, y: int, jitter: int = 3, img_size: int = 224) -> tuple:
    """Add small spatial jitter to avoid exact duplicates."""
    x = int(np.clip(x + random.randint(-jitter, jitter), 0, img_size - 1))
    y = int(np.clip(y + random.randint(-jitter, jitter), 0, img_size - 1))
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Per-object generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_object_samples(obj_name: str, obj_cfg: dict,
                             region_priors: dict,
                             samples_per_cell: int) -> list:
    """
    Each 'cell' = one (region, task) pair.
    Guarantees equal sample count per cell → balanced labels.
    """
    records = []
    tasks = obj_cfg["tasks"]
    regions = obj_cfg["regions"]
    images = obj_cfg["images"]

    for region_name, bbox in regions.items():
        for task_name, task_cfg in tasks.items():
            correct_region = task_cfg["correct_region"]
            label = 1 if region_name == correct_region else 0
            task_id = task_cfg["task_id"]
            action  = task_cfg["action"]

            for _ in range(samples_per_cell):
                x, y = sample_point_in_bbox(bbox)
                x, y = jitter_point(x, y)

                # Verify still inside bbox after jitter (safety)
                if not in_bbox(x, y, bbox):
                    x, y = sample_point_in_bbox(bbox)

                stab = compute_stability(x, y, region_name, region_priors)
                img  = random.choice(images)

                records.append({
                    "object_name":     obj_name,
                    "image":           img,
                    "grasp_x":         x,
                    "grasp_y":         y,
                    "task_id":         task_id,
                    "task_name":       task_name,
                    "action":          action,
                    "label":           label,
                    "region":          region_name,
                    "stability_score": round(stab, 4)
                })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_dataset(records: list) -> bool:
    """
    Runs sanity checks. Returns True if all pass.
    """
    import pandas as pd
    df = pd.DataFrame(records)

    print("\n" + "=" * 50)
    print("DATASET VERIFICATION")
    print("=" * 50)

    passed = True

    # 1. Score range
    out_of_range = (~df["stability_score"].between(0, 1)).sum()
    status = "✓" if out_of_range == 0 else "✗"
    print(f"{status} Stability score range [0,1]: {out_of_range} violations")
    if out_of_range > 0:
        passed = False

    # 2. Label balance per object
    print("\n── Label balance per object ──")
    balance = df.groupby(["object_name", "task_name", "label"]).size().unstack(fill_value=0)
    print(balance)

    # 3. Label–stability correlation (must be low)
    corr = df["label"].corr(df["stability_score"])
    status = "✓" if abs(corr) < 0.35 else "✗"
    print(f"\n{status} Label-stability Pearson correlation: {corr:.4f} (target < 0.35)")
    if abs(corr) >= 0.35:
        passed = False

    # 4. Per-object counts
    print("\n── Per-object sample counts ──")
    print(df.groupby("object_name").size().to_string())

    # 5. Overall stats
    print(f"\n── Overall ──")
    print(f"Total samples : {len(df)}")
    print(f"Positive rate : {df['label'].mean():.3f} (target ≈ 0.50)")
    print(f"Score mean    : {df['stability_score'].mean():.3f}")
    print(f"Score std     : {df['stability_score'].std():.3f}")

    print("\n" + ("✓ All checks passed." if passed else "✗ Some checks FAILED — review config."))
    print("=" * 50 + "\n")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(cfg: dict, output_path: str, seed: int = 42) -> list:
    random.seed(seed)
    np.random.seed(seed)

    dataset_cfg    = cfg["dataset"]
    objects_cfg    = dataset_cfg["objects"]
    region_priors  = dataset_cfg["region_stability_prior"]
    samples_per_cell = dataset_cfg["num_samples_per_cell"]

    all_records = []
    for obj_name, obj_cfg in objects_cfg.items():
        recs = generate_object_samples(
            obj_name, obj_cfg, region_priors, samples_per_cell
        )
        all_records.extend(recs)
        print(f"  {obj_name}: {len(recs)} samples generated")

    random.shuffle(all_records)

    # Stratified split
    split_cfg = dataset_cfg["split"]
    n = len(all_records)
    n_train = int(n * split_cfg["train"])
    n_val   = int(n * split_cfg["val"])

    splits = {
        "train": all_records[:n_train],
        "val":   all_records[n_train:n_train + n_val],
        "test":  all_records[n_train + n_val:]
    }

    os.makedirs(output_path, exist_ok=True)
    for split_name, split_data in splits.items():
        fpath = os.path.join(output_path, f"{split_name}.json")
        with open(fpath, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"  Saved {split_name}: {len(split_data)} samples → {fpath}")

    # Save full dataset for verification
    full_path = os.path.join(output_path, "full_dataset.json")
    with open(full_path, "w") as f:
        json.dump(all_records, f, indent=2)

    verify_dataset(all_records)
    return all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="data/")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"\nGenerating dataset with {cfg['dataset']['num_samples_per_cell']} samples/cell...")
    generate_dataset(cfg, args.output, args.seed)
