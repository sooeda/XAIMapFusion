import warnings
warnings.filterwarnings("ignore")

import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import Config
from utils import ensure_dir, set_seed
from model_and_data import build_dataframe, split_by_patient
from pipeline import (
    VisualSimilarityConsensusMultiTriplet,
    evaluate_classifier_on_test_roi,
    evaluate_localization,
)


@dataclass
class FinalTestFromSavedConfig:
    data_root: str = "kaggle_3m"
    csv_path: str = "slices_labels.csv"
    checkpoint_path: str = "best_tcga_classifier.pth"

    best_configs_path: str = "hyperparam_search_outputs_multitriplet/joint_average_best_per_triplet_val.json"

    output_dir: str = "joint_average_final_test_from_saved_configs"

    seed: int = 42

    fixed_max_initial_regions_per_merged_zone: int = 1

    top_n_triplets_to_run: Optional[int] = None


CFG = FinalTestFromSavedConfig()

METRIC_COLUMNS: Tuple[str, ...] = (
    "dice",
    "iou",
    "precision",
    "recall",
    "pointing_game",
    "relevance_mass",
)
AVERAGE_METHOD_NAME = "JointPixelAverage"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_tp_cases(df: pd.DataFrame, max_cases: Optional[int] = None) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    if max_cases is None:
        max_cases = 30

    if max_cases <= 0 or max_cases >= len(df):
        return df.reset_index(drop=True)

    out = df.sort_values("prob_pos", ascending=False).head(max_cases)
    return out.reset_index(drop=True)


def build_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if len(metrics_df) == 0:
        return pd.DataFrame(columns=["method"])

    agg = metrics_df.groupby("method")[list(METRIC_COLUMNS)].agg(["mean", "std", "median", "count"])
    agg.columns = [f"{c[0]}_{c[1]}" for c in agg.columns]
    agg = agg.reset_index()
    return agg


def cfg_to_compact_jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def triplet_name(perturbation_method: str, gradient_method: str, cam_method: str) -> str:
    return f"{perturbation_method} + {gradient_method} + {cam_method}"


def parse_value_from_file(x: Any) -> Any:
    if x is None:
        return None

    if isinstance(x, np.ndarray):
        return tuple(x.tolist())

    if isinstance(x, list):
        return tuple(x)

    if isinstance(x, tuple):
        return x

    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, np.ndarray):
                return tuple(parsed.tolist())
            if isinstance(parsed, list):
                return tuple(parsed)
            return parsed
        except Exception:
            return s

    try:
        na_flag = pd.isna(x)
        if isinstance(na_flag, (bool, np.bool_)) and na_flag:
            return None
    except Exception:
        pass

    return x


def normalize_best_configs_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "triplet_name" not in df.columns:
        required = ["perturbation_method", "gradient_method", "cam_method"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"В файле с лучшими параметрами не хватает колонок {missing}")
        df["triplet_name"] = df.apply(
            lambda r: triplet_name(str(r["perturbation_method"]), str(r["gradient_method"]), str(r["cam_method"])),
            axis=1,
        )

    known_param_cols = [
        "lime_num_samples",
        "shap_num_samples",
        "gradcam_noise_sigma",
        "layercam_noise_sigma",
        "smoothgrad_samples",
        "smoothgrad_noise_stds",
        "integrated_gradients_n_steps",
        "ensemble_top_positive_fraction",
        "max_initial_regions_per_merged_zone",
        "primary_score",
        "secondary_score",
        "dice_mean",
        "iou_mean",
    ]

    for col in known_param_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_value_from_file)

    group_cols = ["triplet_name", "perturbation_method", "gradient_method", "cam_method"]
    duplicated_triplets = df.duplicated(subset=group_cols, keep=False).any()

    if duplicated_triplets:
        sort_cols = []
        ascending = []

        if "triplet_name" in df.columns:
            sort_cols.append("triplet_name")
            ascending.append(True)

        if "primary_score" in df.columns:
            sort_cols.append("primary_score")
            ascending.append(False)

        if "secondary_score" in df.columns:
            sort_cols.append("secondary_score")
            ascending.append(False)

        if "dice_mean" in df.columns and "dice_mean" not in sort_cols:
            sort_cols.append("dice_mean")
            ascending.append(False)

        if "iou_mean" in df.columns and "iou_mean" not in sort_cols:
            sort_cols.append("iou_mean")
            ascending.append(False)

        if not sort_cols:
            raise ValueError("")

        df = (
            df.sort_values(sort_cols, ascending=ascending)
            .groupby(group_cols, as_index=False)
            .first()
            .reset_index(drop=True)
        )

    final_sort_cols = []
    final_ascending = []
    if "dice_mean" in df.columns:
        final_sort_cols.append("dice_mean")
        final_ascending.append(False)
    if "iou_mean" in df.columns:
        final_sort_cols.append("iou_mean")
        final_ascending.append(False)
    if "primary_score" in df.columns and "primary_score" not in final_sort_cols:
        final_sort_cols.append("primary_score")
        final_ascending.append(False)
    if "secondary_score" in df.columns and "secondary_score" not in final_sort_cols:
        final_sort_cols.append("secondary_score")
        final_ascending.append(False)

    if final_sort_cols:
        df = df.sort_values(final_sort_cols, ascending=final_ascending).reset_index(drop=True)

    return df


def load_best_joint_configs(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Не найден файл {p}")

    suffix = p.suffix.lower()

    if suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Неподдерживаемый формат {p}")

    elif suffix == ".csv":
        df = pd.read_csv(p)

    else:
        raise ValueError(f"Неподдерживаемый формат {p.suffix} ")

    if len(df) == 0:
        raise ValueError(f"Файл {p} пустой")

    df = normalize_best_configs_df(df)

    required_cols = [
        "triplet_name",
        "perturbation_method",
        "gradient_method",
        "cam_method",
        "ensemble_top_positive_fraction",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"В файле {p} не хватает колонок{missing}")

    return df


def build_joint_average_cfg(
    base_cfg,
    perturbation_method: str,
    gradient_method: str,
    cam_method: str,
    param_updates: Dict[str, Any],
    ensemble_top_positive_fraction: float,
    fixed_max_initial_regions_per_merged_zone: int,
):
    return Config(
        **{
            **base_cfg.__dict__,
            "perturbation_method": str(perturbation_method),
            "gradient_method": str(gradient_method),
            "cam_method": str(cam_method),
            "top_positive_fraction": float(ensemble_top_positive_fraction),
            "max_initial_regions_per_merged_zone": int(fixed_max_initial_regions_per_merged_zone),
            "save_qualitative_examples": 0,
            **param_updates,
        }
    )


def evaluate_joint_average_on_selected_tp(
    cfg,
    selected_tp_df: pd.DataFrame,
    triplet_methods: Tuple[str, str, str],
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
            requested_methods=list(triplet_methods),
        )

        maps = []
        for method_name in triplet_methods:
            if method_name not in result["baseline_maps"]:
                raise RuntimeError(f"Неверное название метода {method_name} ")
            maps.append(np.asarray(result["baseline_maps"][method_name], dtype=np.float32))

        avg_map = np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)
        avg_map = runner.positive_importance_map(runner.restrict_to_roi(avg_map, roi_mask))

        metrics = evaluate_localization(
            runner,
            avg_map,
            gt_mask,
            roi_mask,
            pred_mask=None,
        )

        rows.append(
            {
                "image_path": row["image_path"],
                "slice_name": row["slice_name"],
                "method": AVERAGE_METHOD_NAME,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(rows)
    summary_df = build_summary(metrics_df)
    return metrics_df, summary_df


def run_final_test_joint_average_all_triplets(
    base_cfg,
    test_selected_tp_df: pd.DataFrame,
    cfg_from_saved,
    best_per_triplet_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_parts = []
    summary_rows = []

    param_keys = [
        "lime_num_samples",
        "shap_num_samples",
        "gradcam_noise_sigma",
        "layercam_noise_sigma",
        "smoothgrad_samples",
        "smoothgrad_noise_stds",
        "integrated_gradients_n_steps",
    ]

    for _, row in best_per_triplet_df.iterrows():
        param_updates = {}
        for key in param_keys:
            if key in row.index:
                val = parse_value_from_file(row[key])
                if val is not None:
                    if key in {"lime_num_samples", "shap_num_samples", "smoothgrad_samples", "integrated_gradients_n_steps"}:
                        val = int(val)
                    elif key in {"gradcam_noise_sigma", "layercam_noise_sigma"}:
                        val = float(val)
                    elif key == "smoothgrad_noise_stds":
                        if isinstance(val, list):
                            val = tuple(val)
                    param_updates[key] = val

        cfg = build_joint_average_cfg(
            base_cfg=base_cfg,
            perturbation_method=str(row["perturbation_method"]),
            gradient_method=str(row["gradient_method"]),
            cam_method=str(row["cam_method"]),
            param_updates=param_updates,
            ensemble_top_positive_fraction=float(row["ensemble_top_positive_fraction"]),
            fixed_max_initial_regions_per_merged_zone=int(cfg_from_saved.fixed_max_initial_regions_per_merged_zone),
        )

        metrics_df, summary_df = evaluate_joint_average_on_selected_tp(
            cfg=cfg,
            selected_tp_df=test_selected_tp_df,
            triplet_methods=(
                str(row["perturbation_method"]),
                str(row["gradient_method"]),
                str(row["cam_method"]),
            ),
        )

        metrics_df = metrics_df.copy()
        metrics_df["triplet_name"] = str(row["triplet_name"])
        metrics_df["perturbation_method"] = str(row["perturbation_method"])
        metrics_df["gradient_method"] = str(row["gradient_method"])
        metrics_df["cam_method"] = str(row["cam_method"])
        metrics_df["ensemble_top_positive_fraction"] = float(row["ensemble_top_positive_fraction"])

        for k, v in param_updates.items():
            metrics_df[k] = [v] * len(metrics_df)

        metrics_parts.append(metrics_df)

        summary_row = summary_df.iloc[0].to_dict()
        summary_rows.append(
            {
                "triplet_name": str(row["triplet_name"]),
                "perturbation_method": str(row["perturbation_method"]),
                "gradient_method": str(row["gradient_method"]),
                "cam_method": str(row["cam_method"]),
                "ensemble_top_positive_fraction": float(row["ensemble_top_positive_fraction"]),
                **{k: param_updates.get(k, None) for k in param_keys},
                **summary_row,
            }
        )

        print(row["triplet_name"])

    all_metrics_df = pd.concat(metrics_parts, axis=0, ignore_index=True)
    final_summary_df = pd.DataFrame(summary_rows).sort_values(
        ["dice_mean", "iou_mean"],
        ascending=False
    ).reset_index(drop=True)

    return all_metrics_df, final_summary_df


def main(cfg_from_saved: FinalTestFromSavedConfig):
    set_seed(cfg_from_saved.seed)

    output_dir = Path(cfg_from_saved.output_dir)
    ensure_dir(output_dir)

    best_per_triplet_df = load_best_joint_configs(cfg_from_saved.best_configs_path)

    if cfg_from_saved.top_n_triplets_to_run is not None and cfg_from_saved.top_n_triplets_to_run > 0:
        best_per_triplet_df = best_per_triplet_df.head(cfg_from_saved.top_n_triplets_to_run).reset_index(drop=True)

    base_cfg = Config()
    base_cfg.data_root = cfg_from_saved.data_root
    base_cfg.csv_path = cfg_from_saved.csv_path
    base_cfg.checkpoint_path = cfg_from_saved.checkpoint_path
    base_cfg.output_dir = str(output_dir / "base_pipeline_outputs")
    base_cfg.save_qualitative_examples = 0
    base_cfg.max_initial_regions_per_merged_zone = int(cfg_from_saved.fixed_max_initial_regions_per_merged_zone)

    df = build_dataframe(
        Path(base_cfg.data_root),
        Path(base_cfg.csv_path),
        mask_threshold=base_cfg.mask_threshold,
    )

    train_df, val_df, test_df = split_by_patient(df, random_state=cfg_from_saved.seed)

    base_runner = VisualSimilarityConsensusMultiTriplet(base_cfg)
    base_runner.load_model()

    test_pred_df = evaluate_classifier_on_test_roi(base_runner, test_df)

    test_tp_df = test_pred_df[test_pred_df["true_positive"] == 1].copy().reset_index(drop=True)
    test_tp_selected_df = select_tp_cases(test_tp_df)

    if len(test_tp_selected_df) == 0:
        print("нет TP-срезов")
        return

    final_metrics_df, final_summary_df = run_final_test_joint_average_all_triplets(
        base_cfg=base_cfg,
        test_selected_tp_df=test_tp_selected_df,
        cfg_from_saved=cfg_from_saved,
        best_per_triplet_df=best_per_triplet_df,
    )

    final_summary_df.to_csv(output_dir / "joint_average_final_test_summary_all_triplets.csv", index=False)

    with open(output_dir / "joint_average_final_test_summary_all_triplets.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {k: (list(v) if isinstance(v, tuple) else v) for k, v in row.items()}
                for row in final_summary_df.to_dict(orient="records")
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    show_cols = [
        "triplet_name",
        "ensemble_top_positive_fraction",
        "dice_mean",
        "iou_mean",
        "relevance_mass_mean",
        "pointing_game_mean",
    ]
    show_cols = [c for c in show_cols if c in final_summary_df.columns]
    print(final_summary_df[show_cols].to_string(index=False))

    print("\nСохранены файлы")
    print(f"- {output_dir / 'joint_average_final_test_summary_all'}")


if __name__ == "__main__":
    main(CFG)