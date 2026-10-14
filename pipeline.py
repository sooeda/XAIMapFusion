import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from skimage.segmentation import mark_boundaries

from config import ALL_BASELINE_METHODS, CFG, Config
from utils import ensure_dir, normalize01, set_seed
from model_and_data import ModelAndDataMixin, build_dataframe, split_by_patient
from explainers import ExplainersMixin
from consensus import ConsensusMixin
from roi import SelectROI
ROIMixin = SelectROI

class VisualSimilarityConsensusMultiTriplet(
    ConsensusMixin,
    ExplainersMixin,
    ModelAndDataMixin,
    ROIMixin,
):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.saliency = None
        self.integrated_gradients = None
        self.kernel_shap = None
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        self.min_object_size = max(128, self.cfg.image_size * self.cfg.image_size // 80)
        self.min_hole_size = max(64, self.cfg.image_size * self.cfg.image_size // 200)
        self.sample_cache: Dict[str, Dict] = {}
        set_seed(self.cfg.seed)

    def explain_roi_image_once(
        self,
        model,
        image_roi,
        gt_mask,
        roi_mask,
        target_label=1,
        seed_offset=0,
        requested_methods: Optional[List[str]] = None,
    ):
        initial_segments = self.build_initial_segments(image_roi, roi_mask)
        merged_segments, merge_rows = self.merge_segments_by_visual_similarity(
            image_roi, roi_mask, initial_segments
        )
        methods_to_compute = self.resolve_requested_methods(requested_methods)

        baseline_pixel_runs: Dict[str, List[np.ndarray]] = {m: [] for m in methods_to_compute}
        baseline_region_runs: Dict[str, List[np.ndarray]] = {m: [] for m in methods_to_compute}

        if "LIME" in methods_to_compute:
            for r in range(self.cfg.lime_runs):
                pixel_map = self.run_lime_pixel_map(
                    model=model,
                    image_np_for_model=image_roi,
                    fixed_segments=initial_segments,
                    target_label=target_label,
                    seed=self.cfg.seed + seed_offset + 100 + r,
                )
                pixel_map = self.restrict_to_roi(pixel_map, roi_mask)
                pixel_map = self.positive_importance_map(pixel_map)
                baseline_pixel_runs["LIME"].append(pixel_map)
                baseline_region_runs["LIME"].append(self.region_mean(merged_segments, pixel_map))

        if "SHAP" in methods_to_compute:
            for r in range(self.cfg.shap_runs):
                pixel_map = self.run_shap_pixel_map(
                    model=model,
                    image_np_for_model=image_roi,
                    fixed_segments=initial_segments,
                    target_label=target_label,
                    seed=self.cfg.seed + seed_offset + 600 + r,
                )
                pixel_map = self.restrict_to_roi(pixel_map, roi_mask)
                pixel_map = self.positive_importance_map(pixel_map)
                baseline_pixel_runs["SHAP"].append(pixel_map)
                baseline_region_runs["SHAP"].append(self.region_mean(merged_segments, pixel_map))

        if "GradCAM" in methods_to_compute:
            for r in range(self.cfg.gradcam_runs):
                noisy_img = self.add_small_noise_uint8(
                    image_roi,
                    seed=self.cfg.seed + seed_offset + 1100 + r,
                    sigma=self.cfg.gradcam_noise_sigma,
                )
                pixel_map = self.run_gradcam_pixel_map(model, noisy_img, target_label=target_label)
                pixel_map = self.restrict_to_roi(pixel_map, roi_mask)
                pixel_map = self.positive_importance_map(pixel_map)
                baseline_pixel_runs["GradCAM"].append(pixel_map)
                baseline_region_runs["GradCAM"].append(self.region_mean(merged_segments, pixel_map))

        if "LayerCAM" in methods_to_compute:
            for r in range(self.cfg.layercam_runs):
                noisy_img = self.add_small_noise_uint8(
                    image_roi,
                    seed=self.cfg.seed + seed_offset + 1600 + r,
                    sigma=self.cfg.layercam_noise_sigma,
                )
                pixel_map = self.run_layercam_pixel_map(model, noisy_img, target_label=target_label)
                pixel_map = self.restrict_to_roi(pixel_map, roi_mask)
                pixel_map = self.positive_importance_map(pixel_map)
                baseline_pixel_runs["LayerCAM"].append(pixel_map)
                baseline_region_runs["LayerCAM"].append(self.region_mean(merged_segments, pixel_map))

        if "SmoothGrad" in methods_to_compute:
            for r in range(self.cfg.smoothgrad_runs):
                pixel_map = self.run_smoothgrad_pixel_map(
                    model=model,
                    image_np_for_model=image_roi,
                    target_label=target_label,
                    run_idx=r,
                )
                pixel_map = self.restrict_to_roi(pixel_map, roi_mask)
                pixel_map = self.positive_importance_map(pixel_map)
                baseline_pixel_runs["SmoothGrad"].append(pixel_map)
                baseline_region_runs["SmoothGrad"].append(self.region_mean(merged_segments, pixel_map))

        if "IntegratedGradients" in methods_to_compute:
            for r in range(self.cfg.integrated_gradients_runs):
                pixel_map = self.run_integrated_gradients_pixel_map(
                    model=model,
                    image_np_for_model=image_roi,
                    target_label=target_label,
                    run_idx=r,
                )
                pixel_map = self.restrict_to_roi(pixel_map, roi_mask)
                pixel_map = self.positive_importance_map(pixel_map)
                baseline_pixel_runs["IntegratedGradients"].append(pixel_map)
                baseline_region_runs["IntegratedGradients"].append(self.region_mean(merged_segments, pixel_map))

        baseline_maps = {}
        for method_name, runs in baseline_pixel_runs.items():
            if len(runs) == 0:
                baseline_maps[method_name] = np.zeros(image_roi.shape[:2], dtype=np.float32)
            else:
                baseline_maps[method_name] = self.positive_importance_map(
                    np.mean(np.stack(runs, axis=0), axis=0)
                )

        selected_triplet = self.selected_triplet_methods()
        if requested_methods is None or "Proposed_VisualMergeConsensus" in (requested_methods or []):
            consensus_methods = list(selected_triplet)
        else:
            consensus_methods = [
                method_name
                for method_name in methods_to_compute
                if len(baseline_region_runs.get(method_name, [])) > 0
            ]
            if len(consensus_methods) == 0:
                consensus_methods = list(selected_triplet)

        method_runs_region = {
            method_name: (
                np.stack(baseline_region_runs[method_name], axis=0)
                if len(baseline_region_runs.get(method_name, [])) > 0
                else np.zeros((0, 0), dtype=np.float32)
            )
            for method_name in consensus_methods
        }

        faithfulness_scores = self.compute_region_faithfulness(
            model=model,
            image_np_for_model=image_roi,
            segments=merged_segments,
            target_label=target_label,
        )
        combo = self.combine_region_scores(method_runs_region, faithfulness_scores)
        consensus_map = self.region_scores_to_map(merged_segments, combo["consensus"])
        consensus_map = self.restrict_to_roi(consensus_map, roi_mask)

        consensus_input_maps = {}
        for method_name in consensus_methods:
            method_region_scores = combo["method_mean"].get(method_name, np.zeros(0, dtype=np.float32))
            method_map = self.region_scores_to_map(merged_segments, method_region_scores)
            method_map = self.positive_importance_map(self.restrict_to_roi(method_map, roi_mask))
            consensus_input_maps[method_name] = method_map

        agreement_map = self.region_scores_to_map(merged_segments, combo["agreement"])
        stability_map = self.region_scores_to_map(merged_segments, combo["stability"])
        faithfulness_map = self.region_scores_to_map(merged_segments, combo["faithfulness"])
        support_count_map = self.region_scores_to_map(
            merged_segments,
            combo["support_count"].astype(np.float32),
        )
        agreement_gate_map = self.region_scores_to_map(merged_segments, combo["agreement_gate"])
        stability_gate_map = self.region_scores_to_map(merged_segments, combo["stability_gate"])
        importance_gate_map = self.region_scores_to_map(merged_segments, combo["importance_gate"])
        proposed_mask_map = self.region_scores_to_map(merged_segments, combo["proposed_binary_zone_mask"])

        agreement_map = self.positive_importance_map(self.restrict_to_roi(agreement_map, roi_mask))
        stability_map = self.positive_importance_map(self.restrict_to_roi(stability_map, roi_mask))
        faithfulness_map = self.positive_importance_map(self.restrict_to_roi(faithfulness_map, roi_mask))
        support_count_map = self.restrict_to_roi(support_count_map, roi_mask)
        agreement_gate_map = self.positive_importance_map(self.restrict_to_roi(agreement_gate_map, roi_mask))
        stability_gate_map = self.positive_importance_map(self.restrict_to_roi(stability_gate_map, roi_mask))
        importance_gate_map = self.positive_importance_map(self.restrict_to_roi(importance_gate_map, roi_mask))
        proposed_mask_map = self.positive_importance_map(self.restrict_to_roi(proposed_mask_map, roi_mask))

        proposed_binary_mask = self.mask_inside_roi((proposed_mask_map > 0).astype(np.uint8), roi_mask)

        return {
            "image_roi": image_roi,
            "gt_mask": gt_mask,
            "roi_mask": roi_mask,
            "initial_segments": initial_segments,
            "merged_segments": merged_segments,
            "merge_rows": merge_rows,
            "baseline_maps": baseline_maps,
            "consensus_input_maps": consensus_input_maps,
            "selected_triplet": selected_triplet,
            "agreement_map": agreement_map,
            "stability_map": stability_map,
            "faithfulness_map": faithfulness_map,
            "support_count_map": support_count_map,
            "agreement_gate_map": agreement_gate_map,
            "stability_gate_map": stability_gate_map,
            "importance_gate_map": importance_gate_map,
            "consensus_map": self.positive_importance_map(consensus_map),
            "proposed_binary_mask": proposed_binary_mask,
            "consensus_region_scores": combo["consensus"],
            "agreement_region_scores": combo["agreement"],
            "stability_region_scores": combo["stability"],
            "faithfulness_region_scores": combo["faithfulness"],
            "support_count_region": combo["support_count"],
            "n_initial_regions": int(self.n_regions(initial_segments)),
            "n_merged_regions": int(self.n_regions(merged_segments)),
            "agreement_gate_active": int(np.sum(combo["agreement_gate"] > 0)),
            "stability_gate_active": int(np.sum(combo["stability_gate"] > 0)),
            "faithfulness_gate_active": int(np.sum(combo["faithfulness_gate"] > 0)),
            "importance_gate_active": int(np.sum(combo["importance_gate"] > 0)),
        }

    def save_visualization(self, result, out_path):
        import math

        image_np = result["image_roi"]
        gt_mask = result["gt_mask"]
        roi_mask = result["roi_mask"]
        initial_segments = result["initial_segments"]
        merged_segments = result["merged_segments"]
        selected_triplet = result["selected_triplet"]

        panels = [
            ("ROI input + GT mask", None, None),
            ("ROI mask (resized)", None, None),
            (f"Initial regions ({result['n_initial_regions']})", None, None),
            (f"Merged zones ({result['n_merged_regions']})", None, None),
        ]

        for method_name in ALL_BASELINE_METHODS:
            if method_name in result["baseline_maps"]:
                panels.append((f"{method_name} baseline", result["baseline_maps"][method_name], None))

        for method_name in selected_triplet:
            if method_name in result["consensus_input_maps"]:
                panels.append((f"{method_name} inside consensus", result["consensus_input_maps"][method_name], None))

        panels.extend(
            [
                (
                    f"Faithfulness (gate active={result['faithfulness_gate_active']})",
                    result["faithfulness_map"],
                    None,
                ),
                (
                    f"Agreement over triplet (active={result['agreement_gate_active']})",
                    result["agreement_map"],
                    None,
                ),
                (
                    f"Support count map (0..3)",
                    normalize01(result["support_count_map"]),
                    None,
                ),
                (
                    f"Stability (gate active={result['stability_gate_active']})",
                    result["stability_map"],
                    None,
                ),
                (
                    f"Important OR gate (active={result['importance_gate_active']})",
                    result["importance_gate_map"],
                    result["proposed_binary_mask"],
                ),
                (
                    f"Proposed consensus: {selected_triplet[0]} + {selected_triplet[1]} + {selected_triplet[2]}",
                    result["consensus_map"],
                    result["proposed_binary_mask"],
                ),
            ]
        )

        ncols = 4
        nrows = int(math.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
        axes = np.asarray(axes).reshape(-1)

        axes[0].imshow(image_np)
        axes[0].imshow(gt_mask, cmap="Reds", alpha=0.35)
        axes[0].set_title("ROI input + GT mask")
        axes[0].axis("off")

        axes[1].imshow(roi_mask, cmap="gray")
        axes[1].set_title("ROI mask (resized)")
        axes[1].axis("off")

        init_vis = mark_boundaries(
            image_np.astype(np.float32) / 255.0,
            self.segments_for_visualization(initial_segments),
            color=(1, 1, 0),
        )
        axes[2].imshow(init_vis)
        axes[2].set_title(f"Initial regions ({result['n_initial_regions']})")
        axes[2].axis("off")

        merged_vis = mark_boundaries(
            image_np.astype(np.float32) / 255.0,
            self.segments_for_visualization(merged_segments),
            color=(0, 1, 1),
        )
        axes[3].imshow(merged_vis)
        axes[3].set_title(f"Merged zones ({result['n_merged_regions']})")
        axes[3].axis("off")

        for ax, (name, heatmap, custom_mask) in zip(axes[4:], panels[4:]):
            pred_mask = (
                custom_mask
                if custom_mask is not None
                else self.mask_inside_roi(self.top_fraction_mask(heatmap), roi_mask)
            )
            ax.imshow(image_np)
            ax.imshow(heatmap, cmap="jet", alpha=0.45)
            ax.contour(pred_mask, levels=[0.5], colors="cyan", linewidths=1.2)
            ax.contour(gt_mask, levels=[0.5], colors="white", linewidths=1.0)
            ax.set_title(name)
            ax.axis("off")

        for ax in axes[len(panels):]:
            ax.axis("off")

        plt.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


VisualSimilarityConsensusLimeGradCAMSmoothGrad = VisualSimilarityConsensusMultiTriplet


def evaluate_classifier_on_test_roi(
    runner: VisualSimilarityConsensusMultiTriplet,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    images = []
    meta = []

    for _, row in test_df.iterrows():
        sample = runner.prepare_roi_sample(row["image_path"], row["mask_path"])
        images.append(sample["image_roi"])
        meta.append(row)

    probs = runner.predict_proba_numpy(runner.model, images, batch_size=32)
    preds = probs.argmax(axis=1)

    for idx, row in enumerate(meta):
        rows.append(
            {
                "patient_id": row["patient_id"],
                "slice_name": row["slice_name"],
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "label": int(row["label"]),
                "pred": int(preds[idx]),
                "prob_pos": float(probs[idx, 1]),
                "true_positive": int(int(row["label"]) == 1 and int(preds[idx]) == 1),
            }
        )

    return pd.DataFrame(rows)


def select_tp_cases(tp_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if len(tp_df) == 0:
        return tp_df.copy()

    mode = str(cfg.tp_selection_mode).lower()

    if mode == "top_prob":
        selected = tp_df.sort_values("prob_pos", ascending=False).head(cfg.max_tp_cases)
    elif mode == "random":
        selected = tp_df.sample(n=min(cfg.max_tp_cases, len(tp_df)), random_state=cfg.seed)
    else:
        selected = tp_df.head(cfg.max_tp_cases)

    return selected.reset_index(drop=True)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / (denom + 1e-8))


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(inter / (union + 1e-8))


def precision_score_mask(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    if (tp + fp) == 0:
        return 0.0
    return float(tp / (tp + fp + 1e-8))


def recall_score_mask(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    if (tp + fn) == 0:
        return 0.0
    return float(tp / (tp + fn + 1e-8))


def pointing_game_hit(heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
    if heatmap.max() <= 0:
        return 0.0
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return float(gt_mask[y, x] > 0)


def relevance_mass_accuracy(heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
    mass_total = heatmap.sum()
    if mass_total <= 0:
        return 0.0
    return float(heatmap[gt_mask > 0].sum() / (mass_total + 1e-8))


def evaluate_localization(
    runner: VisualSimilarityConsensusMultiTriplet,
    heatmap: np.ndarray,
    gt_mask: np.ndarray,
    roi_mask: np.ndarray,
    pred_mask: np.ndarray = None,
):
    if pred_mask is None:
        pred_mask = runner.top_fraction_mask(heatmap, runner.cfg.top_positive_fraction)
        pred_mask = runner.mask_inside_roi(pred_mask, roi_mask)
    else:
        pred_mask = runner.mask_inside_roi(pred_mask, roi_mask)

    return {
        "dice": dice_score(pred_mask, gt_mask),
        "iou": iou_score(pred_mask, gt_mask),
        "precision": precision_score_mask(pred_mask, gt_mask),
        "recall": recall_score_mask(pred_mask, gt_mask),
        "pointing_game": pointing_game_hit(heatmap, gt_mask),
        "relevance_mass": relevance_mass_accuracy(heatmap, gt_mask),
        "pred_area": int(pred_mask.sum()),
        "gt_area": int(gt_mask.sum()),
    }


def run_localization_on_selected_tp(
    runner: VisualSimilarityConsensusMultiTriplet,
    selected_tp_df: pd.DataFrame,
    output_dir: Path,
):
    loc_rows = []
    region_rows = []
    qual_dir = output_dir / "qualitative_examples"
    ensure_dir(qual_dir)
    saved_examples = 0

    for i, row in selected_tp_df.iterrows():
        sample = runner.prepare_roi_sample(row["image_path"], row["mask_path"])
        image_roi = sample["image_roi"]
        gt_mask = sample["gt_mask"].astype(np.uint8)
        roi_mask = sample["roi_mask"].astype(np.uint8)

        result = runner.explain_roi_image_once(
            model=runner.model,
            image_roi=image_roi,
            gt_mask=gt_mask,
            roi_mask=roi_mask,
            target_label=1,
            seed_offset=10000 * i,
            requested_methods=None,
        )

        base_meta = {
            "patient_id": row["patient_id"],
            "slice_name": row["slice_name"],
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "label": int(row["label"]),
            "pred": int(row["pred"]),
            "prob_pos": float(row["prob_pos"]),
        }

        maps_for_eval = {
            method_name: (result["baseline_maps"][method_name], None)
            for method_name in ALL_BASELINE_METHODS
            if method_name in result["baseline_maps"]
        }
        maps_for_eval["Proposed_VisualMergeConsensus"] = (
            result["consensus_map"],
            result["proposed_binary_mask"],
        )

        for method_name, (heatmap, custom_mask) in maps_for_eval.items():
            metrics = evaluate_localization(runner, heatmap, gt_mask, roi_mask, pred_mask=custom_mask)
            loc_rows.append({**base_meta, "method": method_name, **metrics})

        region_rows.append(
            {
                **base_meta,
                "selected_triplet": " + ".join(result["selected_triplet"]),
                "n_initial_regions": result["n_initial_regions"],
                "n_merged_regions": result["n_merged_regions"],
                "merge_ratio": float(result["n_merged_regions"] / max(result["n_initial_regions"], 1)),
                "agreement_gate_active_regions": int(result["agreement_gate_active"]),
                "stability_gate_active_regions": int(result["stability_gate_active"]),
                "faithfulness_gate_active_regions": int(result["faithfulness_gate_active"]),
                "importance_gate_active_regions": int(result["importance_gate_active"]),
            }
        )

        if saved_examples < runner.cfg.save_qualitative_examples:
            out_path = qual_dir / f"{saved_examples:03d}_{row['slice_name']}.png"
            runner.save_visualization(result, out_path)
            saved_examples += 1

    return pd.DataFrame(loc_rows), pd.DataFrame(region_rows)


def build_localization_summary(loc_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["dice", "iou", "precision", "recall", "pointing_game", "relevance_mass"]

    agg = loc_df.groupby("method")[metrics].agg(["mean", "std", "median", "count"])
    agg.columns = [f"{c[0]}_{c[1]}" for c in agg.columns]
    agg = agg.reset_index()

    order = [
        "LIME",
        "SHAP",
        "SmoothGrad",
        "IntegratedGradients",
        "GradCAM",
        "LayerCAM",
        "Proposed_VisualMergeConsensus",
    ]
    agg["method"] = pd.Categorical(agg["method"], categories=order, ordered=True)
    agg = agg.sort_values("method").reset_index(drop=True)
    return agg


def save_summary_barplots(summary_df: pd.DataFrame, out_path: Path):
    metrics = ["dice", "iou", "relevance_mass", "pointing_game"]

    plt.figure(figsize=(14, 8))
    for i, metric in enumerate(metrics, start=1):
        plt.subplot(2, 2, i)
        plt.bar(summary_df["method"].astype(str), summary_df[f"{metric}_mean"])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(metric)
        plt.title(f"Mean {metric}")
        plt.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(cfg: Config):
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)

    data_root = Path(cfg.data_root)
    csv_path = Path(cfg.csv_path)

    df = build_dataframe(data_root, csv_path, mask_threshold=cfg.mask_threshold)

    train_df, val_df, test_df = split_by_patient(df, random_state=cfg.seed)

    runner = VisualSimilarityConsensusMultiTriplet(cfg)
    runner.load_model()
    test_cls_df = evaluate_classifier_on_test_roi(runner, test_df)
    test_cls_df.to_csv(output_dir / "test_classification_predictions_roi_input.csv", index=False)

    test_acc = 100.0 * (test_cls_df["label"] == test_cls_df["pred"]).mean()

    tp_all_df = test_cls_df[test_cls_df["true_positive"] == 1].copy().reset_index(drop=True)
    tp_all_df.to_csv(output_dir / "test_tp_all_roi_input.csv", index=False)
    if len(tp_all_df) == 0:
        print("")
        return

    tp_selected_df = select_tp_cases(tp_all_df, cfg)
    tp_selected_df.to_csv(output_dir / "test_tp_selected_roi_input.csv", index=False)

    loc_df, region_df = run_localization_on_selected_tp(runner, tp_selected_df, output_dir)
    loc_df.to_csv(output_dir / "localization_metrics_per_slice.csv", index=False)
    region_df.to_csv(output_dir / "region_merge_statistics.csv", index=False)

    if len(loc_df) == 0:
        print("Локализационные метрики не рассчитаны.")
        return

    summary_df = build_localization_summary(loc_df)
    summary_df.to_csv(output_dir / "localization_summary.csv", index=False)
    save_summary_barplots(summary_df, output_dir / "localization_summary_plots.png")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main(CFG)