"""Load fair preprocessed training CSV for fair_CCTC condensation."""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import fair_preprocessed_dataset_dir


class FairCCTC_DataLoaderCreator:
    """
    Reads:
      dataset/preprocessed_datasets_fair/<dataset>/          (default)
      dataset/preprocessed_datasets_fair_op/<dataset>/       (if args.use_op)

    Binary sensitive labels always come from the non-OP train CSV column 'sensitive',
    so fairness reweighting uses the original 0/1 groups even after OP.
    """

    def __init__(self, args):
        self.args = args
        self.use_op = bool(getattr(args, "use_op", False))
        self.fair_variant = "op" if self.use_op else "fair"
        self.preprocessed_dir = fair_preprocessed_dataset_dir(
            _PROJECT_ROOT, self.args.dataset, use_op=self.use_op
        )
        self.base_preprocessed_dir = fair_preprocessed_dataset_dir(
            _PROJECT_ROOT, self.args.dataset, use_op=False
        )
        self.train_csv_path = os.path.join(
            self.preprocessed_dir, f"{self.args.dataset}_train.csv"
        )
        self.sensitive_attr = self._load_sensitive_attr()

    def _load_sensitive_attr(self):
        override = getattr(self.args, "sensitive_attr", None)
        if override:
            return override
        info_path = os.path.join(
            self.base_preprocessed_dir, f"{self.args.dataset}_preprocessed_info.json"
        )
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            if "sensitive_attr" in info:
                return info["sensitive_attr"]
        # Fallback: datasets_info.json
        meta_path = os.path.join(
            _PROJECT_ROOT, "dataset", "download_datasets", "datasets_info.json"
        )
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                all_info = json.load(f)
            key = self.args.dataset
            if key not in all_info:
                by_lower = {k.lower(): k for k in all_info}
                key = by_lower.get(self.args.dataset.lower(), key)
            if key in all_info and "sensitive_attribute" in all_info[key]:
                return all_info[key]["sensitive_attribute"]
        return None

    def load_train(self):
        if not os.path.exists(self.train_csv_path):
            print(f"[Train_data] Missing:\n  {self.train_csv_path}")
            sys.exit(1)
        print(f"[Train_data] Found:\n  {self.train_csv_path}")
        print(f"[Train_data] fair_variant={self.fair_variant}")
        if self.sensitive_attr:
            print(f"[Train_data] sensitive_attr={self.sensitive_attr}")

        df_train = pd.read_csv(self.train_csv_path)
        if "target" not in df_train.columns:
            raise ValueError(
                f"Expected column 'target' in {self.train_csv_path}, got: {list(df_train.columns)}"
            )

        num_classes = int(df_train["target"].nunique())
        feat_cols = [c for c in df_train.columns if c != "target"]
        X_train = df_train[feat_cols].values.astype("float32")
        y_train = df_train["target"].values.astype("int64")

        dst_train = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        return dst_train, num_classes, feat_cols

    def load_base_binary_sensitive_train(self) -> np.ndarray:
        """Binary S from non-OP fair train CSV (aligned row order with OP split)."""
        base_train = os.path.join(
            self.base_preprocessed_dir, f"{self.args.dataset}_train.csv"
        )
        if not os.path.exists(base_train):
            print(f"[Train_data] Missing base sensitive CSV:\n  {base_train}")
            sys.exit(1)
        df = pd.read_csv(base_train)
        if "sensitive" not in df.columns:
            raise ValueError(
                f"Expected column 'sensitive' in {base_train}, got: {list(df.columns)}"
            )
        s = df["sensitive"].astype(int).values
        print(f"[Train_data] fairness S loaded from base train ({len(s)} rows)")
        return s
