"""
Inference module — loads checkpoint, runs GQS evaluation on single or batch inputs.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from models.baseline import ResNetFiLM
from models.physics  import ResNetFiLMPhysics
from utils.metrics   import grasp_quality_score
from dataset.dataloader import generate_synthetic_image


def load_model(ckpt_path: str, model_type: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt["config"]

    if model_type == "baseline":
        model = ResNetFiLM(cfg)
    else:
        model = ResNetFiLMPhysics(cfg)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded {model_type} from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return model, cfg


def preprocess(image: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std= [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # [1, 3, 224, 224]


def encode_grasp(x: int, y: int, img_size: int = 224) -> torch.Tensor:
    gx = (x / img_size) * 2.0 - 1.0
    gy = (y / img_size) * 2.0 - 1.0
    return torch.tensor([[gx, gy]], dtype=torch.float32)  # [1, 2]


def infer_single(model, cfg: dict, image: Image.Image,
                  grasp_x: int, grasp_y: int,
                  task_id: int, model_type: str) -> dict:
    """
    Runs inference on a single grasp candidate.

    Args:
        model     : loaded model (eval mode)
        image     : PIL Image 224×224
        grasp_x/y : pixel coordinates
        task_id   : 0=use, 1=handover
        model_type: "baseline" or "physics"

    Returns:
        result dict with GQS percentage and verdict
    """
    device = next(model.parameters()).device
    inf_cfg = cfg["inference"]

    img_t    = preprocess(image).to(device)
    task_t   = torch.tensor([task_id], dtype=torch.long).to(device)
    grasp_t  = encode_grasp(grasp_x, grasp_y).to(device)

    with torch.no_grad():
        if model_type == "physics":
            logit_cls, pred_stab, pred_physics = model(img_t, task_t, grasp_t)
            p_cls     = torch.sigmoid(logit_cls).item()
            p_stab    = torch.sigmoid(pred_stab).item()
            p_physics = torch.sigmoid(pred_physics).item()
            result    = grasp_quality_score(
                p_cls, p_stab, p_physics,
                alpha=inf_cfg["alpha"],
                beta=inf_cfg["beta"],
                gamma=inf_cfg["gamma"]
            )
        else:
            logit_cls, pred_stab = model(img_t, task_t, grasp_t)
            p_cls  = torch.sigmoid(logit_cls).item()
            p_stab = torch.sigmoid(pred_stab).item()
            result = grasp_quality_score(
                p_cls, p_stab,
                alpha=inf_cfg["alpha"]
            )

    result["grasp_x"]  = grasp_x
    result["grasp_y"]  = grasp_y
    result["task_id"]  = task_id
    return result


def rank_grasp_candidates(model, cfg: dict, image: Image.Image,
                            candidates: list, task_id: int,
                            model_type: str) -> list:
    """
    Rank multiple grasp candidates by GQS.
    candidates: list of (x, y) tuples
    Returns: sorted list of result dicts, best first.
    """
    results = []
    for x, y in candidates:
        r = infer_single(model, cfg, image, x, y, task_id, model_type)
        results.append(r)
    return sorted(results, key=lambda r: r["grasp_quality"], reverse=True)


def demo(model_type: str = "baseline"):
    """Quick demo using synthetic image — no real image needed."""
    ckpt_path = f"checkpoints/{model_type}/best_model.pt"
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}. Train first.")
        return

    model, cfg = load_model(ckpt_path, model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    objects_cfg = cfg["dataset"]["objects"]
    obj_name = "mug"
    regions  = objects_cfg[obj_name]["regions"]

    # Generate synthetic image
    image = generate_synthetic_image(obj_name, regions, 224, augment=False)

    # Sample candidates: handle region vs body region
    candidates = [
        (190, 100),   # handle
        (110, 115),   # body
        (50,  50),    # corner (bad grasp)
        (112, 112),   # center
    ]

    print(f"\n{'='*55}")
    print(f"  GRASP RANKING DEMO — {obj_name} | task=use | {model_type}")
    print(f"{'='*55}")

    ranked = rank_grasp_candidates(model, cfg, image, candidates, task_id=0, model_type=model_type)

    for rank, r in enumerate(ranked, 1):
        print(f"  #{rank} ({r['grasp_x']:3d},{r['grasp_y']:3d}) | "
              f"GQS: {r['grasp_quality']:5.1f}% | "
              f"Sem: {r['semantic_score']:5.1f}% | "
              f"Stab: {r['stability_score']:5.1f}% | "
              f"→ {r['verdict']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "physics"], default="baseline")
    args = parser.parse_args()
    demo(args.model)
