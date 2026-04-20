"""
Generic trainer — handles train loop, val loop, early stopping,
backbone unfreezing, LR scheduling, checkpointing, TensorBoard logging.
Works for both baseline and physics models via duck-typed forward/loss calls.
"""

import os
import time
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

from utils.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_task_breakdown,
    compute_object_breakdown
)
from utils.visualization import (
    plot_confusion_matrix,
    plot_gqs_distribution,
    plot_calibration_curve,
    plot_per_object_accuracy
)
from utils.logger import TrainingLogger


def build_optimizer(model: nn.Module, cfg: dict):
    opt_cfg = cfg["optimizer"]
    if opt_cfg["name"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"]
        )
    elif opt_cfg["name"] == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"]
        )
    raise ValueError(f"Unknown optimizer: {opt_cfg['name']}")


def build_scheduler(optimizer, cfg: dict):
    sched_cfg = cfg["scheduler"]
    name = sched_cfg["name"]
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg["T_max"],
            eta_min=sched_cfg["eta_min"]
        )
    elif name == "step":
        return StepLR(optimizer, step_size=sched_cfg.get("step_size", 20),
                      gamma=sched_cfg.get("gamma", 0.5))
    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    raise ValueError(f"Unknown scheduler: {name}")


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


class Trainer:
    """
    Args:
        model      : ResNetFiLM or ResNetFiLMPhysics
        loss_fn    : BaselineLoss or PhysicsLoss
        loaders    : dict with "train", "val", "test" DataLoaders
        cfg        : full config dict
        model_name : "baseline" or "physics"
        is_physics : bool — controls physics loss gating
    """

    def __init__(self, model, loss_fn, loaders: dict, cfg: dict,
                 model_name: str, is_physics: bool = False):

        self.model      = model
        self.loss_fn    = loss_fn
        self.loaders    = loaders
        self.cfg        = cfg
        self.model_name = model_name
        self.is_physics = is_physics

        train_cfg = cfg["training"]
        self.epochs          = train_cfg["epochs"]
        self.grad_clip       = train_cfg["grad_clip"]
        self.freeze_epochs   = train_cfg["freeze_backbone_epochs"]
        self.physics_warmup  = train_cfg["physics_warmup_epochs"]
        self.inf_cfg         = cfg["inference"]

        self.device = torch.device(
            cfg["project"]["device"]
            if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.optimizer  = build_optimizer(model, cfg)
        self.scheduler  = build_scheduler(self.optimizer, cfg)
        self.early_stop = EarlyStopping(patience=train_cfg["early_stopping_patience"])

        self.logger = TrainingLogger(log_dir="runs/", model_name=model_name)
        self.ckpt_dir = os.path.join("checkpoints", model_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.best_state    = None

        self.logger.info(f"Model: {model_name} | Params: {model.count_parameters():,}")
        self.logger.info(f"Device: {self.device}")

    # ──────────────────────────────────────────────────────────────────────────
    # Forward pass helpers (duck-typed for both models)
    # ──────────────────────────────────────────────────────────────────────────

    def _forward_and_loss(self, batch: dict, apply_physics: bool = True):
        image       = batch["image"].to(self.device)
        task_id     = batch["task_id"].to(self.device)
        grasp_point = batch["grasp_point"].to(self.device)
        label       = batch["label"].to(self.device)
        stability   = batch["stability"].to(self.device)

        if self.is_physics:
            logit_cls, pred_stab, pred_physics = self.model(image, task_id, grasp_point)
            loss, breakdown = self.loss_fn(
                logit_cls, pred_stab, pred_physics,
                label, stability, apply_physics
            )
        else:
            logit_cls, pred_stab = self.model(image, task_id, grasp_point)
            loss, breakdown = self.loss_fn(logit_cls, pred_stab, label, stability)

        return loss, breakdown, torch.sigmoid(logit_cls).detach().cpu().numpy(), \
               torch.sigmoid(pred_stab).detach().cpu().numpy()

    # ──────────────────────────────────────────────────────────────────────────
    # One epoch
    # ──────────────────────────────────────────────────────────────────────────

    def _run_epoch(self, split: str, epoch: int) -> tuple:
        is_train = (split == "train")
        apply_physics = is_train and (epoch >= self.physics_warmup)

        self.model.train() if is_train else self.model.eval()
        loader = self.loaders[split]

        total_loss  = 0.0
        all_labels  = []
        all_probs   = []
        all_stab_t  = []
        all_stab_p  = []
        all_tasks   = []
        all_objects = []
        breakdown_accum = {}

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for batch in loader:
                if is_train:
                    self.optimizer.zero_grad()

                loss, breakdown, probs, pred_stab = self._forward_and_loss(
                    batch, apply_physics
                )

                if is_train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                total_loss += loss.item()
                all_labels.extend(batch["label"].numpy().tolist())
                all_probs.extend(probs.flatten().tolist())
                all_stab_t.extend(batch["stability"].numpy().tolist())
                all_stab_p.extend(pred_stab.flatten().tolist())
                all_tasks.extend(batch["task_name"])
                all_objects.extend(batch["object_name"])

                for k, v in breakdown.items():
                    breakdown_accum[k] = breakdown_accum.get(k, 0) + v

        n_batches = len(loader)
        avg_loss  = total_loss / n_batches
        avg_breakdown = {k: v / n_batches for k, v in breakdown_accum.items()}

        labels  = np.array(all_labels)
        probs   = np.array(all_probs)
        stab_t  = np.array(all_stab_t)
        stab_p  = np.array(all_stab_p)

        cls_metrics  = compute_classification_metrics(labels, probs)
        reg_metrics  = compute_regression_metrics(stab_t, stab_p)
        task_metrics = compute_task_breakdown(labels, probs, all_tasks)
        obj_metrics  = compute_object_breakdown(labels, probs, all_objects)

        return avg_loss, avg_breakdown, cls_metrics, reg_metrics, task_metrics, obj_metrics, \
               labels, probs, stab_p

    # ──────────────────────────────────────────────────────────────────────────
    # Full training loop
    # ──────────────────────────────────────────────────────────────────────────

    def train(self):
        self.model.backbone.freeze_backbone()
        self.logger.info(f"Backbone frozen for first {self.freeze_epochs} epochs.")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # Unfreeze backbone after warmup
            if epoch == self.freeze_epochs + 1:
                self.model.backbone.unfreeze_backbone()
                self.logger.info(f"Epoch {epoch}: Backbone unfrozen.")

            # Physics loss gate
            if self.is_physics and epoch == self.physics_warmup + 1:
                self.logger.info(f"Epoch {epoch}: Physics loss activated.")

            # ── Train ──
            (tr_loss, tr_break, tr_cls, tr_reg,
             tr_task, tr_obj, _, _, _) = self._run_epoch("train", epoch)

            # ── Val ──
            (vl_loss, vl_break, vl_cls, vl_reg,
             vl_task, vl_obj, vl_labels, vl_probs, vl_stab_p) = self._run_epoch("val", epoch)

            # ── Scheduler ──
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(vl_loss)
            else:
                self.scheduler.step()

            # ── Logging ──
            self._log_epoch(epoch, tr_loss, tr_break, tr_cls, tr_reg,
                            vl_loss, vl_break, vl_cls, vl_reg,
                            vl_task, vl_obj)

            # ── Checkpoint ──
            if vl_loss < self.best_val_loss:
                self.best_val_loss = vl_loss
                self.best_state    = copy.deepcopy(self.model.state_dict())
                self._save_checkpoint(epoch, vl_loss)

            elapsed = time.time() - t0
            self.logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train: {tr_loss:.4f} | Val: {vl_loss:.4f} | "
                f"Acc: {vl_cls['accuracy']:.4f} | "
                f"F1: {vl_cls['f1']:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s"
            )

            # ── TensorBoard figures (every 10 epochs) ──
            if epoch % 10 == 0:
                self._log_figures(epoch, vl_labels, vl_probs, vl_stab_p, vl_obj)

            # ── Early stopping ──
            if self.early_stop.step(vl_loss):
                self.logger.info(f"Early stopping at epoch {epoch}.")
                break

        self.logger.info("Training complete. Running final test evaluation...")
        self.model.load_state_dict(self.best_state)
        test_metrics = self.evaluate("test")
        self.logger.close()
        return test_metrics

    def _log_epoch(self, epoch, tr_loss, tr_break, tr_cls, tr_reg,
                   vl_loss, vl_break, vl_cls, vl_reg, vl_task, vl_obj):
        self.logger.log_scalars("train", {"loss": tr_loss, **tr_cls, **tr_reg}, epoch)
        self.logger.log_scalars("val",   {"loss": vl_loss, **vl_cls, **vl_reg}, epoch)
        self.logger.log_scalars("train/breakdown", tr_break, epoch)
        self.logger.log_scalars("val/breakdown",   vl_break, epoch)
        self.logger.log_scalars("val/per_task",    vl_task,  epoch)
        self.logger.log_scalars("val/per_object",  vl_obj,   epoch)
        self.logger.log_scalars("lr", {"lr": self.optimizer.param_groups[0]["lr"]}, epoch)

    def _log_figures(self, epoch, labels, probs, stab_p, object_names):
        preds = (probs >= 0.5).astype(int)
        cm_fig   = plot_confusion_matrix(labels, preds, f"Confusion Matrix — Epoch {epoch}")
        cal_fig  = plot_calibration_curve(labels, probs, f"Calibration — Epoch {epoch}")

        obj_breakdown = compute_object_breakdown(labels, probs, list(object_names))
        obj_fig  = plot_per_object_accuracy(obj_breakdown, f"Per-Object Acc — Epoch {epoch}")

        gqs = self.inf_cfg["alpha"] * probs + (1 - self.inf_cfg["alpha"]) * stab_p
        gqs_fig  = plot_gqs_distribution(gqs, labels, f"GQS Distribution — Epoch {epoch}")

        self.logger.writer.add_figure("val/confusion_matrix", cm_fig,  epoch)
        self.logger.writer.add_figure("val/calibration",      cal_fig, epoch)
        self.logger.writer.add_figure("val/per_object_acc",   obj_fig, epoch)
        self.logger.writer.add_figure("val/gqs_distribution", gqs_fig, epoch)

    def _save_checkpoint(self, epoch: int, val_loss: float):
        path = os.path.join(self.ckpt_dir, f"best_model.pt")
        torch.save({
            "epoch":      epoch,
            "val_loss":   val_loss,
            "model_name": self.model_name,
            "state_dict": self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "config":     self.cfg,
        }, path)

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(self, split: str = "test") -> dict:
        (loss, breakdown, cls_m, reg_m,
         task_m, obj_m, labels, probs, stab_p) = self._run_epoch(split, epoch=0)

        metrics = {
            "loss": loss,
            **cls_m, **reg_m, **task_m, **obj_m
        }

        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"  {split.upper()} EVALUATION — {self.model_name}")
        self.logger.info(f"{'='*50}")
        for k, v in metrics.items():
            self.logger.info(f"  {k:25s}: {v:.4f}")

        return metrics
