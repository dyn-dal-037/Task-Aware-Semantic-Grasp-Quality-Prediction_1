"""
Master entry point for SemGrasp Minor Project.

Usage:
    python run.py --stage dataset
    python run.py --stage train --model baseline
    python run.py --stage train --model physics
    python run.py --stage train --model both
    python run.py --stage tune  --model baseline --trials 20
    python run.py --stage eval
    python run.py --stage demo  --model baseline
    python run.py --stage all   (dataset → train both → eval)
"""

import argparse
import os
import sys
import yaml
import torch


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def stage_dataset(cfg, args):
    from dataset.generate_dataset import generate_dataset
    print("\n── STAGE: Dataset Generation ──")
    generate_dataset(cfg, args.data_dir, cfg["project"]["seed"])


def stage_train(cfg, args, model_type: str):
    from dataset.dataloader import build_dataloaders
    from models.baseline    import ResNetFiLM
    from models.physics     import ResNetFiLMPhysics
    from losses.losses      import BaselineLoss, PhysicsLoss
    from train.trainer      import Trainer

    print(f"\n── STAGE: Train [{model_type}] ──")

    loaders = build_dataloaders(cfg, args.data_dir)

    if model_type == "baseline":
        model   = ResNetFiLM(cfg)
        loss_fn = BaselineLoss(lambda_reg=cfg["losses"]["baseline"]["lambda_reg"])
        trainer = Trainer(model, loss_fn, loaders, cfg,
                          model_name="baseline", is_physics=False)
    else:
        model   = ResNetFiLMPhysics(cfg)
        loss_fn = PhysicsLoss(cfg)
        trainer = Trainer(model, loss_fn, loaders, cfg,
                          model_name="physics", is_physics=True)

    test_metrics = trainer.train()
    print(f"\nFinal test metrics ({model_type}):")
    for k, v in test_metrics.items():
        print(f"  {k:25s}: {v:.4f}")

    return test_metrics


def stage_tune(cfg, args, model_type: str):
    from tune.hyperparameter_tune import run_tuning
    print(f"\n── STAGE: Hyperparameter Tuning [{model_type}] ──")
    run_tuning(args.config, model_type, args.data_dir)


def stage_eval(cfg, args):
    from evaluate.evaluate import run_evaluation
    print("\n── STAGE: Evaluation ──")
    run_evaluation(args.config, args.data_dir)


def stage_demo(cfg, args):
    from inference.inference import demo
    print(f"\n── STAGE: Inference Demo [{args.model}] ──")
    demo(args.model)


def main():
    parser = argparse.ArgumentParser(
        description="SemGrasp Minor — Task-Aware Grasp Quality Prediction"
    )
    parser.add_argument("--stage",   required=True,
                        choices=["dataset", "train", "tune", "eval", "demo", "all"])
    parser.add_argument("--model",   default="both",
                        choices=["baseline", "physics", "both"])
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--data_dir",default="data/")
    parser.add_argument("--trials",  type=int, default=None,
                        help="Number of Optuna trials (tune stage only)")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg["project"]["seed"])

    print(f"\n{'='*55}")
    print(f"  SemGrasp Minor | Stage: {args.stage.upper()}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*55}")

    if args.stage == "dataset":
        stage_dataset(cfg, args)

    elif args.stage == "train":
        models_to_train = (
            ["baseline", "physics"] if args.model == "both" else [args.model]
        )
        for m in models_to_train:
            stage_train(cfg, args, m)

    elif args.stage == "tune":
        models_to_tune = (
            ["baseline", "physics"] if args.model == "both" else [args.model]
        )
        if args.trials:
            cfg["hyperparameter_tune"]["n_trials"] = args.trials
        for m in models_to_tune:
            stage_tune(cfg, args, m)

    elif args.stage == "eval":
        stage_eval(cfg, args)

    elif args.stage == "demo":
        stage_demo(cfg, args)

    elif args.stage == "all":
        stage_dataset(cfg, args)
        stage_train(cfg, args, "baseline")
        stage_train(cfg, args, "physics")
        stage_eval(cfg, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
