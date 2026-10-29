"""Shared constants and helpers: seeds, class list, parameter accounting, IO."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

SEED = 42
NUM_CLASSES = 10
CROP = 64

# EuroSAT RGB class order (identical to torchvision's EuroSAT.classes).
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

NOISE_MODES = ("gaussian", "salt_pepper", "band_dropout")
SEVERITIES = (0.05, 0.15, 0.30)

PARAM_BUDGET_MB = 5.0


def seed_everything(seed: int = SEED) -> None:
    """Make a run reproducible across python, numpy and torch RNGs.

    cuDNN is pinned to deterministic kernels so a GPU run matches a CPU run's
    ordering. Training itself targets CPU, so this mostly matters for the split.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """CPU by default; a GPU is used only if one happens to be present."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Trainable parameter count (BatchNorm running stats are buffers, not params)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameters_to_mb(n_params: int, bytes_per_param: int = 4) -> float:
    """Size of those parameters once written as fp32, in MiB."""
    return n_params * bytes_per_param / (1024 ** 2)


def save_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def human_mb(n_bytes: int) -> float:
    return n_bytes / (1024 ** 2)
