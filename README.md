# SemGrasp Minor — Task-Aware Semantic Grasp Quality Prediction

## Project Structure
```
semgrasp_minor/
├── config/config.yaml          ← All hyperparameters in one place
├── dataset/
│   ├── generate_dataset.py     ← Synthetic dataset generator
│   └── dataloader.py           ← Dataset + DataLoader + procedural images
├── models/
│   ├── film.py                 ← FiLM layer + GraspPointEncoder
│   ├── baseline.py             ← ResNetFiLM (2 heads)
│   └── physics.py              ← ResNetFiLMPhysics (3 heads)
├── losses/losses.py            ← BaselineLoss + PhysicsLoss (all components)
├── train/trainer.py            ← Trainer (early stopping, ckpt, TensorBoard)
├── evaluate/evaluate.py        ← Full eval + comparison table + plots
├── inference/inference.py      ← GQS scoring + grasp ranking
├── tune/hyperparameter_tune.py ← Optuna TPE tuning
├── utils/
│   ├── metrics.py              ← All metrics (cls + reg + per-task/object)
│   ├── visualization.py        ← Plots (CM, calibration, GQS dist, per-obj)
│   └── logger.py               ← TensorBoard + file logger
├── run.py                      ← Master entry point
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Workflow

### 1. Generate Dataset
```bash
python run.py --stage dataset
# Produces data/train.json, data/val.json, data/test.json
# Runs verification: balance, correlation, score range
```

### 2. Train
```bash
# Baseline only
python run.py --stage train --model baseline

# Physics only
python run.py --stage train --model physics

# Both sequentially
python run.py --stage train --model both

# Full pipeline (dataset + both models + eval)
python run.py --stage all
```

### 3. Monitor Training
```bash
tensorboard --logdir runs/
# Open http://localhost:6006
# Tabs: scalars (loss/acc/f1), figures (CM, calibration, GQS), hparams
```

### 4. Hyperparameter Tuning (Optional — run before final training)
```bash
python run.py --stage tune --model baseline --trials 30
python run.py --stage tune --model physics  --trials 30
# Best params saved to tune/best_params_{model}.json
# Update config/config.yaml manually with best params, then retrain
```

### 5. Evaluate Both Models
```bash
python run.py --stage eval
# Prints comparison table: Baseline vs Physics
# Saves plots to evaluate/
```

### 6. Inference / Grasp Ranking Demo
```bash
python run.py --stage demo --model baseline
python run.py --stage demo --model physics
```

## Colab Workflow
```python
# Mount drive
from google.colab import drive
drive.mount('/content/drive')

# Clone / copy project to Colab
import subprocess
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# Run
!python run.py --stage all --data_dir /content/drive/MyDrive/semgrasp/data/
```

## Key Design Decisions

### Stability Score (No Label Leakage)
```
stability = 0.5 * geometric(x, y) + 0.5 * region_prior
```
Task modifier is NOT included — it only determines the binary label.
Label-stability Pearson correlation should be < 0.35.

### Physics Loss (Gated)
Physics loss is only applied after `physics_warmup_epochs` (default: 5).
This prevents the physics head from destabilizing early training.

### Grasp Quality Score (GQS)
```
Baseline: GQS = 0.60 * p_semantic + 0.40 * p_stability
Physics:  GQS = 0.50 * p_semantic + 0.25 * p_stability + 0.15 * p_physics  [normalized]
```

### Backbone Freezing
ResNet is frozen for first `freeze_backbone_epochs` (default: 5).
This forces FiLM and heads to learn meaningful projections before fine-tuning backbone.

## Expected Results (3-5K dataset)
| Metric    | Baseline  | Physics   |
|-----------|-----------|-----------|
| Accuracy  | ~82–87%   | ~85–90%   |
| F1        | ~0.82–0.87| ~0.85–0.90|
| AUROC     | ~0.88–0.93| ~0.90–0.95|
| Spearman  | ~0.55–0.70| ~0.60–0.75|

Physics model expected to show +2–5% on accuracy and better calibration.
