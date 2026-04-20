"""
Logging utilities: TensorBoard + console.
"""

import os
import json
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def get_run_name(model_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{ts}"


class TrainingLogger:
    """
    Wraps TensorBoard SummaryWriter + Python logger.
    Usage:
        logger = TrainingLogger("runs/", "baseline")
        logger.log_scalars("train", {"loss": 0.4, "acc": 0.88}, epoch=1)
        logger.close()
    """

    def __init__(self, log_dir: str, model_name: str):
        self.run_name = get_run_name(model_name)
        self.run_dir  = os.path.join(log_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.history = {}

        # Console logger
        self.logger = logging.getLogger(self.run_name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            fh = logging.FileHandler(os.path.join(self.run_dir, "train.log"))
            fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
            ch.setFormatter(fmt); fh.setFormatter(fmt)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        self.logger.info(f"Run: {self.run_name}")
        self.logger.info(f"Logs: {self.run_dir}")

    def log_scalars(self, prefix: str, metrics: dict, step: int):
        """Log dict of scalars to TensorBoard and history."""
        for k, v in metrics.items():
            tag = f"{prefix}/{k}"
            self.writer.add_scalar(tag, v, step)
            if tag not in self.history:
                self.history[tag] = []
            self.history[tag].append((step, v))

    def log_loss_breakdown(self, breakdown: dict, step: int, prefix: str = "train"):
        """Log physics loss breakdown dict."""
        self.log_scalars(prefix, breakdown, step)

    def log_hparams(self, hparams: dict, metrics: dict):
        """Log hyperparameters to TensorBoard HParams tab."""
        self.writer.add_hparams(hparams, metrics)

    def log_text(self, tag: str, text: str, step: int):
        self.writer.add_text(tag, text, step)

    def info(self, msg: str):
        self.logger.info(msg)

    def save_history(self):
        path = os.path.join(self.run_dir, "history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def close(self):
        self.save_history()
        self.writer.close()
        self.logger.info("Logger closed.")
