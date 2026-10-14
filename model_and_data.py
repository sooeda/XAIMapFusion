from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models

from sklearn.model_selection import train_test_split

try:
    from captum.attr import Saliency, IntegratedGradients, KernelShap
except ImportError as e:
    raise ImportError("Не найден Captum. Установите зависимости: pip install captum") from e

from config import Config
from utils import has_lesion, pil_bilinear


def build_dataframe(data_root: Path, csv_path: Path, mask_threshold: int = 50) -> pd.DataFrame:
    rows = []

    if not data_root.exists():
        raise FileNotFoundError(f"Папка не найдена: {data_root.resolve()}")

    for patient_dir in sorted(data_root.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        tif_files = sorted(
            [p for p in patient_dir.iterdir() if p.is_file() and p.suffix.lower() == ".tif"],
            key=lambda p: p.name.lower(),
        )

        for img_path in tif_files:
            stem = img_path.stem
            if stem.lower().endswith("_mask"):
                continue

            mask_path = patient_dir / f"{stem}_mask{img_path.suffix}"
            label_value = has_lesion(mask_path, threshold=mask_threshold)

            rows.append(
                {
                    "patient_id": patient_id,
                    "image_path": str(img_path.relative_to(data_root)),
                    "mask_path": str(mask_path.relative_to(data_root)) if mask_path.exists() else "",
                    "label": label_value,
                    "slice_name": stem,
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("Не найдено ни одного MRI-среза")

    df.to_csv(csv_path, index=False)
    return df


def split_by_patient(df: pd.DataFrame, random_state: int = 42):
    patient_stats = df.groupby("patient_id")["label"].agg(["mean", "count"]).reset_index()
    patient_stats["patient_class"] = (patient_stats["mean"] > 0).astype(int)

    unique_classes = patient_stats["patient_class"].value_counts()
    can_stratify = unique_classes.min() >= 2 if len(unique_classes) > 1 else False

    if can_stratify:
        train_patients, temp_patients = train_test_split(
            patient_stats["patient_id"],
            test_size=0.4,
            random_state=random_state,
            stratify=patient_stats["patient_class"],
        )

        temp_stats = patient_stats[patient_stats["patient_id"].isin(temp_patients)].copy()

        val_patients, test_patients = train_test_split(
            temp_stats["patient_id"],
            test_size=0.5,
            random_state=random_state,
            stratify=temp_stats["patient_class"],
        )
    else:
        print(" ")
        train_patients, temp_patients = train_test_split(
            patient_stats["patient_id"],
            test_size=0.4,
            random_state=random_state,
        )
        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=0.5,
            random_state=random_state,
        )

    train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_patients)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)
    return train_df, val_df, test_df


class TCGAClassifier(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = models.resnet18(pretrained=True)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class ModelAndDataMixin:
    cfg: Config
    device: torch.device
    model: nn.Module
    saliency: Saliency
    integrated_gradients: IntegratedGradients
    kernel_shap: KernelShap
    imagenet_mean: np.ndarray
    imagenet_std: np.ndarray
    min_object_size: int
    min_hole_size: int
    sample_cache: Dict[str, Dict]

    def load_model(self):
        checkpoint_path = Path(self.cfg.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Не найден файл весов {checkpoint_path}\n"

            )

        model = TCGAClassifier(num_classes=2).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        self.model = model
        self.saliency = Saliency(self.model)
        self.integrated_gradients = IntegratedGradients(self.model)
        self.kernel_shap = KernelShap(self._kernel_shap_forward)
        return model

    def preprocess_images_np(self, images_np):
        if isinstance(images_np, list):
            images_np = np.stack(images_np, axis=0)
        if images_np.ndim == 3:
            images_np = images_np[None, ...]

        processed = []
        for img in images_np:
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            if arr.shape[:2] != (self.cfg.image_size, self.cfg.image_size):
                arr = np.array(
                    Image.fromarray(arr).resize((self.cfg.image_size, self.cfg.image_size), pil_bilinear()),
                    dtype=np.uint8,
                )
            processed.append(arr.astype(np.float32) / 255.0)

        batch = np.stack(processed, axis=0)
        batch = np.transpose(batch, (0, 3, 1, 2))
        batch = (batch - self.imagenet_mean) / self.imagenet_std
        return torch.from_numpy(batch.astype(np.float32))

    @torch.no_grad()
    def predict_proba_numpy(self, model, images_np, batch_size=32):
        batch_tensor = self.preprocess_images_np(images_np)
        probs = []
        for i in range(0, len(batch_tensor), batch_size):
            batch = batch_tensor[i:i + batch_size].to(self.device)
            logits = model(batch)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
        return np.concatenate(probs, axis=0)

    def load_original_image_np(self, rel_image_path):
        rel_image_path = Path(rel_image_path)
        img_path = Path(self.cfg.data_root) / rel_image_path
        if not img_path.exists():
            raise FileNotFoundError(f"Не найдено изображение: {img_path}")
        img = Image.open(img_path).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def load_original_gt_mask_np(self, rel_mask_path):
        if rel_mask_path is None or rel_mask_path == "":
            return None
        mask_path = Path(self.cfg.data_root) / rel_mask_path
        if not mask_path.exists():
            return None
        mask = Image.open(mask_path).convert("L")
        return (np.array(mask, dtype=np.uint8) > self.cfg.mask_threshold).astype(np.uint8)