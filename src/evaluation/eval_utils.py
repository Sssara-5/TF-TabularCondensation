"""Fairness metrics (ΔDP / ΔEO). Only used for the fair pipeline."""
import os

import numpy as np
import pandas as pd

from utils import fair_preprocessed_dataset_dir


def compute_dp(y_pred, s):
    """
    Demographic Parity gap:
    ΔDP = |P(Ŷ=1|S=0) - P(Ŷ=1|S=1)|
    """
    y_pred = np.asarray(y_pred).astype(np.int32)
    s = np.asarray(s).astype(np.int32)

    mask0 = s == 0
    mask1 = s == 1
    if mask0.sum() == 0 or mask1.sum() == 0:
        return float("nan"), float("nan"), float("nan")

    pr_s0 = float(y_pred[mask0].mean())
    pr_s1 = float(y_pred[mask1].mean())
    delta_dp = abs(pr_s0 - pr_s1)
    return delta_dp, pr_s0, pr_s1


def compute_eo(y_pred, y_true, s):
    """
    Equalized Odds gap (TPR gap on Y=1):
    ΔEO = |P(Ŷ=1|Y=1,S=0) - P(Ŷ=1|Y=1,S=1)|
    """
    y_pred = np.asarray(y_pred).astype(np.int32)
    y_true = np.asarray(y_true).astype(np.int32)
    s = np.asarray(s).astype(np.int32)

    mask_pos0 = (y_true == 1) & (s == 0)
    mask_pos1 = (y_true == 1) & (s == 1)
    if mask_pos0.sum() == 0 or mask_pos1.sum() == 0:
        return float("nan"), float("nan"), float("nan")

    tpr_s0 = float(y_pred[mask_pos0].mean())
    tpr_s1 = float(y_pred[mask_pos1].mean())
    delta_eo = abs(tpr_s0 - tpr_s1)
    return delta_eo, tpr_s0, tpr_s1


def compute_fairness(y_pred, y_true, s):
    """Return ΔDP / ΔEO dict for binary predictions and binary sensitive labels."""
    y_pred = np.asarray(y_pred).astype(np.int32)
    y_true = np.asarray(y_true).astype(np.int32)
    s = np.asarray(s).astype(np.int32)

    delta_dp, pr_s0, pr_s1 = compute_dp(y_pred, s)
    delta_eo, tpr_s0, tpr_s1 = compute_eo(y_pred, y_true, s)
    return {
        "delta_dp": float(delta_dp),
        "pr_s0": float(pr_s0),
        "pr_s1": float(pr_s1),
        "delta_eo": float(delta_eo),
        "tpr_s0": float(tpr_s0),
        "tpr_s1": float(tpr_s1),
    }


def load_base_binary_sensitive(
    project_root,
    dataset_name,
    split="test",
    sensitive_col="sensitive",
):
    """
    Binary S for DP/EO always from non-OP fair preprocess CSV.

    After OP the feature column 'sensitive' is projected (not 0/1). Fairness
    evaluation must use the base file:
      dataset/preprocessed_datasets_fair/<dataset>/<dataset>_{split}.csv
    """
    base_dir = fair_preprocessed_dataset_dir(project_root, dataset_name, use_op=False)
    path = os.path.join(base_dir, f"{dataset_name}_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Base fair {split} CSV for DP/EO not found: {path}. "
            "Run fair_preprocessor first."
        )
    df = pd.read_csv(path)
    if sensitive_col not in df.columns:
        raise ValueError(
            f"Expected column {sensitive_col!r} in {path}, got: {list(df.columns)}"
        )
    s = df[sensitive_col].astype(int).values
    uniq = set(np.unique(s).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(
            f"Sensitive column in {path} is not binary 0/1: unique={sorted(uniq)}"
        )
    return s
