import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from config import CAM_METHODS, GRADIENT_METHODS, PERTURBATION_METHODS, Config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def pil_bilinear():
    return Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def pil_nearest():
    return Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def has_lesion(mask_path: Path, threshold: int = 50) -> int:
    if not mask_path.exists():
        return 0
    try:
        mask = np.array(Image.open(mask_path).convert("L"))
        return int(np.any(mask > threshold))
    except Exception:
        return 0


def selected_triplet_from_cfg(cfg: Config) -> Tuple[str, str, str]:
    if cfg.perturbation_method not in PERTURBATION_METHODS:
        raise ValueError(f"perturbation_method должен быть одним из {PERTURBATION_METHODS}, получено: {cfg.perturbation_method}")
    if cfg.gradient_method not in GRADIENT_METHODS:
        raise ValueError(f"gradient_method должен быть одним из {GRADIENT_METHODS}, получено: {cfg.gradient_method}")
    if cfg.cam_method not in CAM_METHODS:
        raise ValueError(f"cam_method должен быть одним из {CAM_METHODS}, получено: {cfg.cam_method}")
    return cfg.perturbation_method, cfg.gradient_method, cfg.cam_method
