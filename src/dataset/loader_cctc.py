"""Load preprocessed training CSV for CCTC condensation."""
import os
import sys

import pandas as pd
import torch
from torch.utils.data import TensorDataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import preprocessed_dataset_dir


class CCTC_DataLoaderCreator:
    def __init__(self, args):
        self.args = args
        base = preprocessed_dataset_dir(_PROJECT_ROOT, self.args.categorical_method, self.args.dataset)
        self.train_csv_path = os.path.join(base, f"{self.args.dataset}_train.csv")

    def load_train(self):
        if not os.path.exists(self.train_csv_path):
            print(f"[Train_data] Missing:\n  {self.train_csv_path}")
            sys.exit(1)
        print(f"[Train_data] Found:\n  {self.train_csv_path}")
        df_train = pd.read_csv(self.train_csv_path)

        num_classes = len(df_train["target"].unique())
        feat_cols = list(df_train.columns[:-1])
        X_train = df_train[feat_cols].values
        y_train = df_train["target"].values

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        dst_train = TensorDataset(X_train_tensor, y_train_tensor)
        return dst_train, num_classes, feat_cols
