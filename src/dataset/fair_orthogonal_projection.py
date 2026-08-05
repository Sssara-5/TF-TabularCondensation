"""
Orthogonal Projection for Fair-CCTC preprocessing outputs.

Input expected column order:
    sensitive + feature columns + target

This script removes one linear sensitive direction from feature space:
    Z_fair = Z - (Z @ unit_bias)[:, None] * unit_bias[None, :]

Supported orthogonal projection method (op_method):
    1. op

Important:
    - Binary sensitive labels are used only to learn the projection direction.
    - All columns except target are projected (including sensitive).
    - target is never projected.
    - For DP/EO evaluation after OP, load binary S from the base (non-OP) CSV.
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


SUPPORTED_METHODS = {"op"}
OP_EPS = 1e-8


def _unit_vector(v, eps=OP_EPS):
    """L2-normalize v."""
    return v / (np.linalg.norm(v) + eps)


def orth_proj(x, sens_idx, eps=OP_EPS):
    """
    Orthogonal projection (numpy port of orth_proj).

    Groups rows by x[:, sens_idx] in {0, 1}, builds unit bias direction from
    normalized group sums, then removes the component along that direction:

        debias_x = x - (x @ unit_bias)[:, None] * unit_bias[None, :]
    """
    x = np.asarray(x, dtype=np.float64)
    groups = x[:, sens_idx]

    idx_zero = np.where(groups == 0)[0]
    idx_one = np.where(groups == 1)[0]
    if len(idx_zero) == 0 or len(idx_one) == 0:
        raise ValueError("OP orth_proj requires both sensitive groups 0 and 1.")

    x_0 = x[idx_zero]
    x_1 = x[idx_one]

    v_0 = x_0.sum(axis=0)
    v_1 = x_1.sum(axis=0)

    unit_v0 = _unit_vector(v_0, eps)
    unit_v1 = _unit_vector(v_1, eps)

    bias_v = unit_v0 - unit_v1
    unit_bias = _unit_vector(bias_v, eps)

    ip_bias = x @ unit_bias
    return x - np.outer(ip_bias, unit_bias), unit_bias


def _safe_auc(y_true, score):
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, score))
    except Exception:
        return None


def _sensitive_predictability(X, A, random_state=42):
    """Train a quick logistic model to measure how much sensitive info remains."""
    if len(np.unique(A)) < 2:
        return {"acc": None, "auc": None}

    clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state)
    clf.fit(X, A)
    pred = clf.predict(X)
    score = clf.predict_proba(X)[:, 1]
    return {
        "acc": float(accuracy_score(A, pred)),
        "auc": _safe_auc(A, score),
    }


def _mean_gap(X, A):
    """Return ||mean(X|A=1) - mean(X|A=0)||."""
    if len(np.unique(A)) != 2:
        return None
    mu0 = X[A == 0].mean(axis=0)
    mu1 = X[A == 1].mean(axis=0)
    return float(np.linalg.norm(mu1 - mu0))


class OrthogonalProjector:
    def __init__(
        self,
        sensitive_col="sensitive",
        target_col="target",
        op_method="op",
        random_state=42,
    ):
        op_method = op_method.lower().strip()
        if op_method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown op_method='{op_method}'. Choose from {sorted(SUPPORTED_METHODS)}"
            )

        self.sensitive_col = sensitive_col
        self.target_col = target_col
        self.op_method = op_method
        self.random_state = random_state

        self.project_cols = None
        self.feature_cols = None
        self.w = None
        self.w_norm_sq = None
        self.sens_idx = None
        self.report = {}

    def fit(self, df_train):
        self.project_cols = [
            c for c in df_train.columns
            if c != self.target_col
        ]
        self.feature_cols = [
            c for c in self.project_cols
            if c != self.sensitive_col
        ]

        if self.sensitive_col not in df_train.columns:
            raise ValueError(f"Missing sensitive column: {self.sensitive_col}")
        if self.target_col not in df_train.columns:
            raise ValueError(f"Missing target column: {self.target_col}")
        if not self.project_cols:
            raise ValueError("No columns to project.")

        A = df_train[self.sensitive_col].astype(int).values
        if len(np.unique(A)) != 2:
            raise ValueError("Orthogonal projection currently expects binary sensitive attribute 0/1.")

        X = df_train[self.project_cols].astype(float).values
        self.sens_idx = self.project_cols.index(self.sensitive_col)
        self._fit_op_direction(X, self.sens_idx)
        X_proj = self._project_op(X)

        self.w_norm_sq = float(np.dot(self.w, self.w))
        if self.w_norm_sq <= OP_EPS ** 2:
            raise ValueError("OP sensitive direction has near-zero norm. Projection is not meaningful.")

        self.report["op_method"] = self.op_method
        self.report["sensitive_direction_type"] = self.op_method
        self.report["projection_includes_sensitive"] = True
        self.report["uses_scaler"] = False
        self.report["op_sens_idx"] = self.sens_idx
        self.report["train_sensitive_predictability_before"] = _sensitive_predictability(
            X, A, self.random_state
        )
        self.report["train_sensitive_predictability_after"] = _sensitive_predictability(
            X_proj, A, self.random_state
        )
        self.report["train_mean_gap_before"] = _mean_gap(X, A)
        self.report["train_mean_gap_after"] = _mean_gap(X_proj, A)
        self.report["num_project_cols"] = len(self.project_cols)
        self.report["num_features"] = len(self.feature_cols)
        self.report["project_cols"] = self.project_cols
        self.report["feature_cols"] = self.feature_cols
        self.report["w_norm"] = float(np.sqrt(self.w_norm_sq))

        return self

    def _fit_op_direction(self, X, sens_idx):
        """
        Learn OP unit_bias from train data (orth_proj direction only).

        Direction matches:
            v_a = sum(x | S=a), unit_va = v_a / ||v_a||
            unit_bias = unit(unit_v0 - unit_v1)
        """
        _, unit_bias = orth_proj(X, sens_idx, eps=OP_EPS)
        self.w = unit_bias.reshape(-1).astype(float)

    def _project_op(self, X):
        """Apply OP projection with fitted unit_bias (no StandardScaler)."""
        ip_bias = X @ self.w
        return X - np.outer(ip_bias, self.w)

    def transform(self, df):
        if self.project_cols is None or self.w is None:
            raise RuntimeError("Projector is not fitted yet.")

        out = df.copy()
        X = out[self.project_cols].astype(float).values
        X_proj = self._project_op(X)
        out[self.project_cols] = X_proj
        out[self.target_col] = out[self.target_col].astype(int)
        return out

    def fit_transform(self, df_train):
        self.fit(df_train)
        return self.transform(df_train)

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "sensitive_direction_w.npy"), self.w)

        with open(os.path.join(out_dir, "projection_report.json"), "w") as f:
            json.dump(self.report, f, indent=4)


def run_projection(
    input_dir,
    output_dir=None,
    dataset_name=None,
    sensitive_attr=None,
    sensitive_col="sensitive",
    target_col="target",
    op_method="op",
    random_state=42,
):
    op_method = op_method.lower().strip()
    if op_method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown op_method='{op_method}'. Choose from {sorted(SUPPORTED_METHODS)}"
        )

    if output_dir is None:
        # Mirror standard leaf layout:
        #   input:  .../preprocessed_datasets_fair/<dataset>/
        #   output: .../preprocessed_datasets_fair_op/<dataset>/
        base = input_dir.rstrip(os.sep)
        parent = os.path.dirname(base)
        dataset_leaf = os.path.basename(base)
        output_dir = os.path.join(f"{parent}_op", dataset_leaf)

    if dataset_name is None:
        train_candidates = [f for f in os.listdir(input_dir) if f.endswith("_train.csv")]
        if not train_candidates:
            raise FileNotFoundError(f"No *_train.csv found in {input_dir}")
        dataset_name = train_candidates[0].replace("_train.csv", "")

    train_path = os.path.join(input_dir, f"{dataset_name}_train.csv")
    val_path = os.path.join(input_dir, f"{dataset_name}_val.csv")
    test_path = os.path.join(input_dir, f"{dataset_name}_test.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    projector = OrthogonalProjector(
        sensitive_col=sensitive_col,
        target_col=target_col,
        op_method=op_method,
        random_state=random_state,
    )

    df_train_fair = projector.fit_transform(df_train)
    df_val_fair = projector.transform(df_val)
    df_test_fair = projector.transform(df_test)

    os.makedirs(output_dir, exist_ok=True)
    df_train_fair.to_csv(os.path.join(output_dir, f"{dataset_name}_train.csv"), index=False)
    df_val_fair.to_csv(os.path.join(output_dir, f"{dataset_name}_val.csv"), index=False)
    df_test_fair.to_csv(os.path.join(output_dir, f"{dataset_name}_test.csv"), index=False)

    projector.save(output_dir)

    info_path = os.path.join(input_dir, f"{dataset_name}_preprocessed_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
    else:
        info = {}

    base_subdir = os.path.basename(input_dir.rstrip(os.sep))
    info.update({
        "orthogonal_projection": True,
        "projection_op_method": op_method,
        "projection_includes_sensitive": True,
        "projection_sensitive_col": sensitive_col,
        "projection_target_col": target_col,
        "projection_project_cols": projector.project_cols,
        "projection_feature_cols": projector.feature_cols,
        "projection_report": projector.report,
        "fairness_sensitive_source": "base",
        "fairness_sensitive_col": sensitive_col,
        "fairness_sensitive_col_index": 0,
        "base_preprocessed_subdir": base_subdir,
    })
    if sensitive_attr is not None:
        info["sensitive_attr"] = sensitive_attr

    with open(os.path.join(output_dir, f"{dataset_name}_preprocessed_info.json"), "w") as f:
        json.dump(info, f, indent=4)

    print(f"[Done] Fair projected files saved to: {output_dir}")
    print(f"[op_method] {op_method}")
    print("[Report] Sensitive predictability before projection:", projector.report["train_sensitive_predictability_before"])
    print("[Report] Sensitive predictability after projection:", projector.report["train_sensitive_predictability_after"])
    print("[Report] Mean gap before projection:", projector.report["train_mean_gap_before"])
    print("[Report] Mean gap after projection:", projector.report["train_mean_gap_after"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing <dataset>_train.csv, <dataset>_val.csv, <dataset>_test.csv")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Default: <parent>_op/<dataset> "
                             "(e.g. preprocessed_datasets_fair_op/<dataset>/)")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset name prefix. If omitted, inferred from *_train.csv")
    parser.add_argument("--sensitive_attr", type=str, default=None)
    parser.add_argument("--sensitive_col", type=str, default="sensitive")
    parser.add_argument("--target_col", type=str, default="target")
    parser.add_argument("--op_method", type=str, default="op", choices=sorted(SUPPORTED_METHODS),
                        help="Orthogonal projection method used to learn sensitive direction w.")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    run_projection(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        sensitive_attr=args.sensitive_attr,
        sensitive_col=args.sensitive_col,
        target_col=args.target_col,
        op_method=args.op_method,
        random_state=args.random_state,
    )
