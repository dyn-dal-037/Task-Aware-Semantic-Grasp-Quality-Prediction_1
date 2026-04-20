"""
Hyperparameter tuning with Optuna (TPE sampler + Median pruner).
Tunes both baseline and physics models.
Run: python tune/hyperparameter_tune.py --model baseline --trials 30
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import yaml
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner

from dataset.dataloader import build_dataloaders
from models.baseline    import ResNetFiLM
from models.physics     import ResNetFiLMPhysics
from losses.losses      import BaselineLoss, PhysicsLoss
from train.trainer      import Trainer


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────

def build_objective(base_cfg: dict, model_type: str, data_dir: str):

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)
        ss  = base_cfg["hyperparameter_tune"]["search_space"]

        # ── Sample hyperparameters ──
        lr             = trial.suggest_float("lr",             *ss["lr"],             log=True)
        weight_decay   = trial.suggest_float("weight_decay",   *ss["weight_decay"],   log=True)
        dropout        = trial.suggest_float("dropout",        *ss["dropout"])
        lambda_reg     = trial.suggest_float("lambda_reg",     *ss["lambda_reg"])
        batch_size     = trial.suggest_categorical("batch_size", ss["batch_size"])
        task_embed_dim = trial.suggest_categorical("task_embed_dim", ss["task_embed_dim"])

        # Apply to config
        cfg["optimizer"]["lr"]          = lr
        cfg["optimizer"]["weight_decay"] = weight_decay
        cfg["model"]["dropout"]          = dropout
        cfg["model"]["task_embed_dim"]   = task_embed_dim
        cfg["training"]["batch_size"]    = batch_size
        cfg["losses"]["baseline"]["lambda_reg"] = lambda_reg

        if model_type == "physics":
            lambda_physics = trial.suggest_float("lambda_physics", *ss["lambda_physics"])
            cfg["losses"]["physics"]["lambda_reg"]     = lambda_reg
            cfg["losses"]["physics"]["lambda_physics"] = lambda_physics

        # Reduce epochs for tuning
        cfg["training"]["epochs"]                   = 20
        cfg["training"]["early_stopping_patience"]  = 5
        cfg["training"]["freeze_backbone_epochs"]   = 2
        cfg["training"]["physics_warmup_epochs"]    = 3

        # ── Build components ──
        try:
            loaders = build_dataloaders(cfg, data_dir)

            if model_type == "baseline":
                model   = ResNetFiLM(cfg)
                loss_fn = BaselineLoss(lambda_reg=lambda_reg)
                trainer = Trainer(model, loss_fn, loaders, cfg,
                                  model_name=f"tune_baseline_t{trial.number}",
                                  is_physics=False)
            else:
                model   = ResNetFiLMPhysics(cfg)
                loss_fn = PhysicsLoss(cfg)
                trainer = Trainer(model, loss_fn, loaders, cfg,
                                  model_name=f"tune_physics_t{trial.number}",
                                  is_physics=True)

            # Abbreviated training loop with pruning
            best_val_loss = float("inf")
            for epoch in range(1, cfg["training"]["epochs"] + 1):
                trainer.model.train()
                for batch in loaders["train"]:
                    trainer.optimizer.zero_grad()
                    loss, _, _, _ = trainer._forward_and_loss(
                        batch, apply_physics=(epoch >= cfg["training"]["physics_warmup_epochs"])
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
                    trainer.optimizer.step()

                # Val
                val_loss, _, val_cls, _, _, _, _, _, _ = trainer._run_epoch("val", epoch)
                val_acc = val_cls["accuracy"]

                trial.report(val_loss, epoch)
                if trial.should_prune():
                    trainer.logger.close()
                    raise optuna.exceptions.TrialPruned()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            trainer.logger.close()
            return best_val_loss

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return float("inf")

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_tuning(cfg_path: str, model_type: str, data_dir: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    tune_cfg = cfg["hyperparameter_tune"]

    study = optuna.create_study(
        direction = "minimize",
        sampler   = TPESampler(seed=cfg["project"]["seed"]),
        pruner    = MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name= f"semgrasp_{model_type}"
    )

    study.optimize(
        build_objective(cfg, model_type, data_dir),
        n_trials  = tune_cfg["n_trials"],
        timeout   = tune_cfg["timeout"],
        n_jobs    = 1,    # Set >1 only if you have multiple GPUs
        show_progress_bar = True
    )

    print("\n" + "=" * 50)
    print(f"BEST TRIAL ({model_type})")
    print("=" * 50)
    print(f"  Val loss : {study.best_trial.value:.4f}")
    print("  Params   :")
    for k, v in study.best_trial.params.items():
        print(f"    {k:20s}: {v}")

    # Save best params
    os.makedirs("tune", exist_ok=True)
    import json
    with open(f"tune/best_params_{model_type}.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)

    print(f"\nBest params saved to tune/best_params_{model_type}.json")
    return study.best_trial.params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config/config.yaml")
    parser.add_argument("--model",    choices=["baseline", "physics"], default="baseline")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--trials",   type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.trials:
        cfg["hyperparameter_tune"]["n_trials"] = args.trials

    run_tuning(args.config, args.model, args.data_dir)
