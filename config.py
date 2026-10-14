from dataclasses import dataclass
from typing import Tuple

PERTURBATION_METHODS: Tuple[str, ...] = ("LIME", "SHAP")
GRADIENT_METHODS: Tuple[str, ...] = ("SmoothGrad", "IntegratedGradients")
CAM_METHODS: Tuple[str, ...] = ("GradCAM", "LayerCAM")
ALL_BASELINE_METHODS: Tuple[str, ...] = (
    *PERTURBATION_METHODS,
    *GRADIENT_METHODS,
    *CAM_METHODS,
)


@dataclass
class Config:
    data_root: str = "kaggle_3m"
    csv_path: str = "slices_labels.csv"
    checkpoint_path: str = "best_tcga_classifier.pth"
    output_dir: str = "visual_similarity_consensus_multitriplet_outputs_hard_rule"
    image_size: int = 224
    seed: int = 42
    mask_threshold: int = 50
    roi_margin: int = 4
    roi_threshold_quantile_mix: Tuple[float, float] = (0.60, 0.80)
    max_tp_cases: int = 30
    tp_selection_mode: str = "top_prob"
    initial_n_segments: int = 50
    initial_compactness: float = 0.18
    initial_sigma: float = 1.0
    merge_similarity_threshold: float = 0.80
    merge_similarity_sigma: float = 0.45
    merge_w_mean: float = 0.60
    merge_w_std: float = 0.15
    merge_w_grad: float = 0.25
    max_initial_regions_per_merged_zone: int = 1
    perturbation_method: str = "LIME"
    gradient_method: str = "SmoothGrad"
    cam_method: str = "GradCAM"
    lime_runs: int = 3
    shap_runs: int = 3
    gradcam_runs: int = 3
    layercam_runs: int = 3
    smoothgrad_runs: int = 3
    integrated_gradients_runs: int = 3
    lime_num_samples: int = 300
    shap_num_samples: int = 200
    gradcam_noise_sigma: float = 2.0
    layercam_noise_sigma: float = 2.0
    smoothgrad_noise_stds: Tuple[float, float, float] = (0.10, 0.12, 0.15)
    smoothgrad_samples: int = 16
    integrated_gradients_n_steps: int = 32
    integrated_gradients_baseline_modes: Tuple[str, str, str] = ("zero", "mean", "blur")
    integrated_gradients_blur_sigma: float = 1.5
    weight_agreement: float = 1.0
    weight_stability: float = 1.0
    weight_faithfulness: float = 1.0
    region_support_top_fraction: float = 0.10
    agreement_min_supported_methods: int = 2
    faithfulness_importance_threshold: float = 0.15
    stability_importance_threshold: float = 0.20
    top_positive_fraction: float = 0.10
    save_qualitative_examples: int = 20


CFG = Config()
