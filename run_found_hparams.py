import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from config import Config
from utils import ensure_dir, set_seed
from model_and_data import build_dataframe, split_by_patient
from pipeline import (
    VisualSimilarityConsensusMultiTriplet,
    evaluate_classifier_on_test_roi,
    evaluate_localization,
    build_localization_summary,
)


DATA_ROOT = "kaggle_3m"
CSV_PATH = "slices_labels.csv"
CHECKPOINT_PATH = "best_tcga_classifier.pth"
SEARCH_OUTPUT_DIR = "hyperparam_search_outputs_multitriplet"
OUTPUT_DIR = "run_found_hparams_outputs"

SEED = 42

EVAL_SPLIT = "test"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_tp_cases(df: pd.DataFrame, max_cases: int = None) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    if max_cases is None:
        max_cases = 30

    out = df.sort_values("prob_pos", ascending=False).head(max_cases)
    return out.reset_index(drop=True)


def evaluate_config_on_selected_tp(
    cfg: Config,
    selected_tp_df: pd.DataFrame,
    methods_to_eval: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runner = VisualSimilarityConsensusMultiTriplet(cfg)
    runner.load_model()

    rows = []

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
            requested_methods=methods_to_eval,
        )

        maps_for_eval = {
            method_name: (result["baseline_maps"][method_name], None)
            for method_name in result["baseline_maps"].keys()
        }
        maps_for_eval["Proposed_VisualMergeConsensus"] = (
            result["consensus_map"],
            result["proposed_binary_mask"],
        )

        for method_name in methods_to_eval:
            heatmap, custom_mask = maps_for_eval[method_name]
            metrics = evaluate_localization(runner, heatmap, gt_mask, roi_mask, pred_mask=custom_mask)
            rows.append(
                {
                    "image_path": row["image_path"],
                    "slice_name": row["slice_name"],
                    "method": method_name,
                    **metrics,
                }
            )

    metrics_df = pd.DataFrame(rows)
    summary_df = build_localization_summary(metrics_df)
    return metrics_df, summary_df


def make_base_cfg(output_dir: Path) -> Config:
    cfg = Config()
    cfg.data_root = DATA_ROOT
    cfg.csv_path = CSV_PATH
    cfg.checkpoint_path = CHECKPOINT_PATH
    cfg.output_dir = str(output_dir)
    cfg.seed = SEED
    cfg.max_tp_cases = 30
    cfg.save_qualitative_examples = 0
    return cfg


def apply_baseline_hparams(base_cfg: Config, best_baseline_params: Dict[str, Dict[str, Any]], method_name: str) -> Config:
    cfg = Config(**base_cfg.__dict__)

    if method_name == "LIME":
        cfg.lime_num_samples = int(best_baseline_params["LIME"]["lime_num_samples"])
        cfg.top_positive_fraction = float(best_baseline_params["LIME"]["top_positive_fraction"])

    elif method_name == "SHAP":
        cfg.shap_num_samples = int(best_baseline_params["SHAP"]["shap_num_samples"])
        cfg.top_positive_fraction = float(best_baseline_params["SHAP"]["top_positive_fraction"])

    elif method_name == "GradCAM":
        cfg.gradcam_noise_sigma = float(best_baseline_params["GradCAM"]["gradcam_noise_sigma"])
        cfg.top_positive_fraction = float(best_baseline_params["GradCAM"]["top_positive_fraction"])

    elif method_name == "LayerCAM":
        cfg.layercam_noise_sigma = float(best_baseline_params["LayerCAM"]["layercam_noise_sigma"])
        cfg.top_positive_fraction = float(best_baseline_params["LayerCAM"]["top_positive_fraction"])

    elif method_name == "SmoothGrad":
        cfg.smoothgrad_samples = int(best_baseline_params["SmoothGrad"]["smoothgrad_samples"])
        cfg.smoothgrad_noise_stds = tuple(best_baseline_params["SmoothGrad"]["smoothgrad_noise_stds"])
        cfg.top_positive_fraction = float(best_baseline_params["SmoothGrad"]["top_positive_fraction"])

    elif method_name == "IntegratedGradients":
        cfg.integrated_gradients_n_steps = int(best_baseline_params["IntegratedGradients"]["integrated_gradients_n_steps"])
        cfg.top_positive_fraction = float(best_baseline_params["IntegratedGradients"]["top_positive_fraction"])

    else:
        raise ValueError(f"Неверное название {method_name}")

    return cfg


def build_proposed_cfg(
    base_cfg: Config,
    best_baseline_params: Dict[str, Dict[str, Any]],
    proposed_row: Dict[str, Any],
) -> Config:
    cfg = Config(**base_cfg.__dict__)

    cfg.perturbation_method = str(proposed_row["perturbation_method"])
    cfg.gradient_method = str(proposed_row["gradient_method"])
    cfg.cam_method = str(proposed_row["cam_method"])

    cfg.lime_num_samples = int(best_baseline_params["LIME"]["lime_num_samples"])
    cfg.shap_num_samples = int(best_baseline_params["SHAP"]["shap_num_samples"])
    cfg.gradcam_noise_sigma = float(best_baseline_params["GradCAM"]["gradcam_noise_sigma"])
    cfg.layercam_noise_sigma = float(best_baseline_params["LayerCAM"]["layercam_noise_sigma"])
    cfg.smoothgrad_samples = int(best_baseline_params["SmoothGrad"]["smoothgrad_samples"])
    cfg.smoothgrad_noise_stds = tuple(best_baseline_params["SmoothGrad"]["smoothgrad_noise_stds"])
    cfg.integrated_gradients_n_steps = int(best_baseline_params["IntegratedGradients"]["integrated_gradients_n_steps"])

    cfg.region_support_top_fraction = float(proposed_row["region_support_top_fraction"])
    cfg.faithfulness_importance_threshold = float(proposed_row["faithfulness_importance_threshold"])
    cfg.stability_importance_threshold = float(proposed_row["stability_importance_threshold"])
    cfg.top_positive_fraction = float(proposed_row["top_positive_fraction"])
    cfg.max_initial_regions_per_merged_zone = int(proposed_row["max_initial_regions_per_merged_zone"])

    return cfg


def main():
    set_seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    ensure_dir(output_dir)

    search_output_dir = Path(SEARCH_OUTPUT_DIR)
    baseline_json_path = search_output_dir / "best_baseline_hyperparameters.json"
    proposed_json_path = search_output_dir / "best_proposed_per_triplet_val.json"

    best_baseline_params = load_json(baseline_json_path)
    best_proposed_per_triplet = load_json(proposed_json_path)

    base_cfg = make_base_cfg(output_dir / "base_pipeline_outputs")
    df = build_dataframe(Path(base_cfg.data_root), Path(base_cfg.csv_path), mask_threshold=base_cfg.mask_threshold)
    train_df, val_df, test_df = split_by_patient(df, random_state=SEED)

    split_df = test_df if EVAL_SPLIT.lower() == "test" else val_df
    split_name = "test" if EVAL_SPLIT.lower() == "test" else "val"

    base_runner = VisualSimilarityConsensusMultiTriplet(base_cfg)
    base_runner.load_model()

    pred_df = evaluate_classifier_on_test_roi(base_runner, split_df)
    tp_df = pred_df[pred_df["true_positive"] == 1].copy().reset_index(drop=True)

    selected_tp_df = select_tp_cases(tp_df)

    if len(selected_tp_df) == 0:
        print("Нет TP-срезов")
        return

    baseline_methods = ["LIME", "SHAP", "GradCAM", "LayerCAM", "SmoothGrad", "IntegratedGradients"]

    baseline_metrics_parts = []
    baseline_summary_rows = []

    for method_name in baseline_methods:
        print(method_name)
        cfg = apply_baseline_hparams(base_cfg, best_baseline_params, method_name)
        metrics_df, summary_df = evaluate_config_on_selected_tp(cfg, selected_tp_df, [method_name])

        metrics_df.to_csv(output_dir / f"found_{method_name.lower()}_metrics_per_slice.csv", index=False)
        summary_df.to_csv(output_dir / f"found_{method_name.lower()}_summary.csv", index=False)

        baseline_metrics_parts.append(metrics_df)
        baseline_summary_rows.append(summary_df)

    final_baseline_metrics_df = pd.concat(baseline_metrics_parts, axis=0, ignore_index=True)
    final_baseline_summary_df = build_localization_summary(final_baseline_metrics_df)

    final_baseline_metrics_df.to_csv(output_dir / "final_found_baseline_metrics_per_slice.csv", index=False)
    final_baseline_summary_df.to_csv(output_dir / "final_found_baseline_summary.csv", index=False)

    proposed_metrics_parts = []
    proposed_summary_rows = []

    for row in best_proposed_per_triplet:
        triplet_name = str(row["triplet_name"])
        print(triplet_name)

        cfg = build_proposed_cfg(base_cfg, best_baseline_params, row)
        metrics_df, summary_df = evaluate_config_on_selected_tp(
            cfg,
            selected_tp_df,
            ["Proposed_VisualMergeConsensus"],
        )

        metrics_df = metrics_df.copy()
        metrics_df["triplet_name"] = triplet_name
        metrics_df["perturbation_method"] = str(row["perturbation_method"])
        metrics_df["gradient_method"] = str(row["gradient_method"])
        metrics_df["cam_method"] = str(row["cam_method"])
        metrics_df["region_support_top_fraction"] = float(row["region_support_top_fraction"])
        metrics_df["faithfulness_importance_threshold"] = float(row["faithfulness_importance_threshold"])
        metrics_df["stability_importance_threshold"] = float(row["stability_importance_threshold"])
        metrics_df["top_positive_fraction"] = float(row["top_positive_fraction"])
        metrics_df["max_initial_regions_per_merged_zone"] = int(row["max_initial_regions_per_merged_zone"])

        proposed_metrics_parts.append(metrics_df)

        summary_row = summary_df.iloc[0].to_dict()
        proposed_summary_rows.append(
            {
                "triplet_name": triplet_name,
                "perturbation_method": str(row["perturbation_method"]),
                "gradient_method": str(row["gradient_method"]),
                "cam_method": str(row["cam_method"]),
                "region_support_top_fraction": float(row["region_support_top_fraction"]),
                "faithfulness_importance_threshold": float(row["faithfulness_importance_threshold"]),
                "stability_importance_threshold": float(row["stability_importance_threshold"]),
                "top_positive_fraction": float(row["top_positive_fraction"]),
                "max_initial_regions_per_merged_zone": int(row["max_initial_regions_per_merged_zone"]),
                **summary_row,
            }
        )

    final_proposed_summary_df = pd.DataFrame(proposed_summary_rows).sort_values(
        ["dice_mean", "iou_mean"],
        ascending=False,
    ).reset_index(drop=True)

    final_proposed_summary_df.to_csv(output_dir / "final_found_proposed_summary_all_triplets.csv", index=False)

    print(final_baseline_summary_df.to_string(index=False))
    print(
        final_proposed_summary_df[
            [
                "triplet_name",
                "region_support_top_fraction",
                "faithfulness_importance_threshold",
                "stability_importance_threshold",
                "top_positive_fraction",
                "max_initial_regions_per_merged_zone",
                "dice_mean",
                "iou_mean",
                "relevance_mass_mean",
                "pointing_game_mean",
            ]
        ].to_string(index=False)
    )

    print("\nСохранены файлы")
    print(f"- {output_dir / 'final_found_baseline_summary.csv'}")
    print(f"- {output_dir / 'final_found_proposed_summary_all_triplets.csv'}")


if __name__ == "__main__":
    main()