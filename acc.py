import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from config import Config
from utils import set_seed
from model_and_data import build_dataframe, split_by_patient
from pipeline import VisualSimilarityConsensusMultiTriplet, evaluate_classifier_on_test_roi


DATA_ROOT = "kaggle_3m"
CSV_PATH = "slices_labels.csv"
CHECKPOINT_PATH = "best_tcga_classifier.pth"
SEED = 42
EVAL_SPLIT = "test"


def build_runner() -> VisualSimilarityConsensusMultiTriplet:
    cfg = Config()
    cfg.data_root = DATA_ROOT
    cfg.csv_path = CSV_PATH
    cfg.checkpoint_path = CHECKPOINT_PATH
    cfg.seed = SEED
    cfg.save_qualitative_examples = 0

    runner = VisualSimilarityConsensusMultiTriplet(cfg)
    runner.load_model()
    return runner


def main():
    set_seed(SEED)

    runner = build_runner()

    df = build_dataframe(
        Path(DATA_ROOT),
        Path(CSV_PATH),
        mask_threshold=runner.cfg.mask_threshold,
    )

    train_df, val_df, test_df = split_by_patient(df, random_state=SEED)

    split_df = test_df if EVAL_SPLIT.lower() == "test" else val_df
    split_name = "test" if EVAL_SPLIT.lower() == "test" else "val"

    pred_df = evaluate_classifier_on_test_roi(runner, split_df)

    y_true = pred_df["label"].astype(int).values
    y_pred = pred_df["pred"].astype(int).values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Split: {split_name}")
    print(f"Num samples: {len(pred_df)}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()