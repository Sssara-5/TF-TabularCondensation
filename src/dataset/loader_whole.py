"""Load full real train/val/test splits for whole-dataset evaluation (same preprocess tree as CCTC)."""
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import preprocessed_dataset_dir


class DataLoaderCreator:
    """Paths: dataset/preprocessed_datasets/<categorical_method>/<dataset>/."""

    def __init__(self, args):
        self.args = args
        base_path = preprocessed_dataset_dir(_PROJECT_ROOT, self.args.categorical_method, self.args.dataset)
        self.train_csv_path = os.path.join(base_path, f"{self.args.dataset}_train.csv")
        self.val_csv_path = os.path.join(base_path, f"{self.args.dataset}_val.csv")
        self.test_csv_path = os.path.join(base_path, f"{self.args.dataset}_test.csv")
        self.info_json_path = os.path.join(base_path, f"{self.args.dataset}_preprocessed_info.json")

    def load_data(self):
        if not os.path.exists(self.train_csv_path):
            print(f"[Train_data] Missing:\n  {self.train_csv_path}")
            sys.exit(1)
        print(f"[Train_data] Found:\n  {self.train_csv_path}")
        df_train = pd.read_csv(self.train_csv_path)

        if not os.path.exists(self.val_csv_path):
            print(f"[Val_data] Missing:\n  {self.val_csv_path}")
            sys.exit(1)
        print(f"[Val_data] Found:\n  {self.val_csv_path}")
        df_val = pd.read_csv(self.val_csv_path)

        if not os.path.exists(self.test_csv_path):
            print(f"[Test_data] Missing:\n  {self.test_csv_path}")
            sys.exit(1)
        print(f"[Test_data] Found:\n  {self.test_csv_path}")
        df_test = pd.read_csv(self.test_csv_path)

        if not os.path.exists(self.info_json_path):
            print(f"[Info] Missing:\n  {self.info_json_path}")
            sys.exit(1)
        with open(self.info_json_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        print(f"[Info] Loaded:\n  {self.info_json_path}")

        attr_name = info.get("attr_name", list(df_train.columns[:-1]))
        train_attr = df_train.columns[:-1]
        X_train = df_train[train_attr].values
        y_train = df_train["target"].values

        val_attr = df_val.columns[:-1]
        X_val = df_val[val_attr].values
        y_val = df_val["target"].values

        test_attr = df_test.columns[:-1]
        X_test = df_test[test_attr].values
        y_test = df_test["target"].values

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        num_classes = len(df_train["target"].unique())

        dst_train = TensorDataset(X_train_tensor, y_train_tensor)
        dst_val = TensorDataset(X_val_tensor, y_val_tensor)
        dst_test = TensorDataset(X_test_tensor, y_test_tensor)

        trainloader = DataLoader(dst_train, batch_size=self.args.batch_train, shuffle=True, drop_last=True)
        valloader = DataLoader(dst_val, batch_size=self.args.batch_train, shuffle=False)
        testloader = DataLoader(dst_test, batch_size=self.args.batch_train, shuffle=False)

        numerical_feature_count = info.get("numerical_feature_count", 0)
        numerical_feature_idx = info.get("numerical_feature_idx", [])
        categorical_feature_count = info.get("categorical_feature_count", 0)
        categorical_feature_idx = info.get("categorical_feature_idx", [])
        unique_values_per_categorical_feature = list(info.get("unique_values_per_categorical_feature", {}).values())

        return (
            trainloader,
            valloader,
            testloader,
            num_classes,
            attr_name,
            numerical_feature_count,
            numerical_feature_idx,
            categorical_feature_count,
            categorical_feature_idx,
            unique_values_per_categorical_feature,
        )
