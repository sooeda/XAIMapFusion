from typing import Dict

import numpy as np
from PIL import Image

from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_holes,
    remove_small_objects,
)

from config import Config
from utils import pil_bilinear, pil_nearest


class SelectROI:
    cfg: Config
    min_object_size: int
    min_hole_size: int
    sample_cache: Dict[str, Dict]

    # Эти методы приходят из ModelAndDataMixin
    def load_original_image_np(self, rel_image_path):
        raise NotImplementedError

    def load_original_gt_mask_np(self, rel_mask_path):
        raise NotImplementedError

    def to_gray_float(self, image_np):
        arr = np.asarray(image_np).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    def largest_connected_component(self, mask):
        mask = np.asarray(mask).astype(bool)
        lab = label(mask.astype(np.uint8), connectivity=2)
        if lab.max() == 0:
            return mask.astype(np.uint8)
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        return (lab == np.argmax(counts)).astype(np.uint8)

    def robust_roi_threshold(self, gray):
        vals = gray[gray > 0]
        if len(vals) == 0:
            return 0.0
        try:
            otsu_thr = float(threshold_otsu(vals))
        except Exception:
            otsu_thr = float(np.quantile(vals, 0.5))
        q20 = float(np.quantile(vals, 0.20))
        a, b = self.cfg.roi_threshold_quantile_mix
        thr = max(a * otsu_thr, b * q20)
        return float(thr)

    def build_brain_roi_mask(self, image_np):
        gray = self.to_gray_float(image_np)
        thr = self.robust_roi_threshold(gray)
        roi = gray > thr
        roi = binary_closing(roi, footprint=disk(5))
        roi = binary_opening(roi, footprint=disk(3))
        roi = remove_small_objects(roi, min_size=self.min_object_size)
        roi = ndi.binary_fill_holes(roi)
        roi = remove_small_holes(roi.astype(bool), area_threshold=self.min_hole_size)
        roi = self.largest_connected_component(roi)
        if roi.sum() == 0:
            roi = np.ones_like(gray, dtype=np.uint8)
        return roi.astype(np.uint8)

    def mask_bbox(self, mask, margin=None):
        margin = self.cfg.roi_margin if margin is None else int(margin)
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return 0, mask.shape[0], 0, mask.shape[1]
        y0 = max(0, int(ys.min()) - margin)
        y1 = min(mask.shape[0], int(ys.max()) + margin + 1)
        x0 = max(0, int(xs.min()) - margin)
        x1 = min(mask.shape[1], int(xs.max()) + margin + 1)
        return y0, y1, x0, x1

    def prepare_roi_sample(self, rel_image_path: str, rel_mask_path: str):
        cache_key = f"{rel_image_path}|||{rel_mask_path}"
        if cache_key in self.sample_cache:
            return self.sample_cache[cache_key]

        image_full = self.load_original_image_np(rel_image_path)
        gt_full = self.load_original_gt_mask_np(rel_mask_path)
        if gt_full is None:
            gt_full = np.zeros(image_full.shape[:2], dtype=np.uint8)

        roi_mask_full = self.build_brain_roi_mask(image_full)
        y0, y1, x0, x1 = self.mask_bbox(roi_mask_full)

        image_crop = image_full[y0:y1, x0:x1]
        gt_crop = gt_full[y0:y1, x0:x1]
        roi_crop = roi_mask_full[y0:y1, x0:x1]

        image_roi = np.array(
            Image.fromarray(image_crop).resize((self.cfg.image_size, self.cfg.image_size), pil_bilinear()),
            dtype=np.uint8,
        )
        gt_roi = np.array(
            Image.fromarray((gt_crop * 255).astype(np.uint8)).resize((self.cfg.image_size, self.cfg.image_size), pil_nearest()),
            dtype=np.uint8,
        )
        roi_mask = np.array(
            Image.fromarray((roi_crop * 255).astype(np.uint8)).resize((self.cfg.image_size, self.cfg.image_size), pil_nearest()),
            dtype=np.uint8,
        )

        sample = {
            "image_roi": image_roi,
            "gt_mask": (gt_roi > 0).astype(np.uint8),
            "roi_mask": (roi_mask > 0).astype(np.uint8),
            "bbox": (x0, y0, x1, y1),
        }
        self.sample_cache[cache_key] = sample
        return sample