"""Load CCTC synthetic CSVs plus matching preprocessed val/test and info JSON."""
import glob
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import cctc_synthetic_output_dir, preprocessed_dataset_dir


class SynDataLoaderCreator:
    def __init__(self, args):
        self.args = args
        if self.args.method != "cctc":
            print(f"[SynDataLoader] Expected args.method=='cctc', got {self.args.method!r}.")
            sys.exit(1)

        self.syn_dataset_folder = cctc_synthetic_output_dir(
            _PROJECT_ROOT,
            self.args.dataset,
            self.args.categorical_method,
            self.args.reduction_rate,
            self.args.gamma,
        )
        self.base_path = preprocessed_dataset_dir(
            _PROJECT_ROOT, self.args.categorical_method, self.args.dataset
        )
        self.val_csv_path = os.path.join(self.base_path, f"{self.args.dataset}_val.csv")
        self.test_csv_path = os.path.join(self.base_path, f"{self.args.dataset}_test.csv")
        self.info_json_path = os.path.join(self.base_path, f"{self.args.dataset}_preprocessed_info.json")

    def load_syn_data(self):
        seed_dfs = {}
        trainloader_list = []

        for seed in range(self.args.num_exp):
            matches = glob.glob(os.path.join(self.syn_dataset_folder, f"*seed{seed}*.csv"))
            if not matches:
                print(f"[Syn_data] Not Found Synthetic Data CSV for seed{seed} in:\n  {self.syn_dataset_folder}")
                sys.exit(1)
            print(f"[Syn_data] Found Synthetic Data CSV for seed{seed}:\n  {matches}")
            seed_dfs[seed] = pd.read_csv(matches[0])

        if os.path.exists(self.val_csv_path):
            print(f"[Val_data] Found Validation Data CSV:\n  {self.val_csv_path}")
            df_val = pd.read_csv(self.val_csv_path)
        else:
            print(f"[Val_data] Not Found Validation Data CSV:\n  {self.val_csv_path}")
            sys.exit(1)

        if os.path.exists(self.test_csv_path):
            print(f"[Test] Found Test Data CSV:\n  {self.test_csv_path}")
            df_test = pd.read_csv(self.test_csv_path)
        else:
            print(f"[Test] Not Found Test Data CSV:\n  {self.test_csv_path}")
            sys.exit(1)

        if os.path.exists(self.info_json_path):
            with open(self.info_json_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            print(f"[Info] Loaded preprocessed info from:\n  {self.info_json_path}")
        else:
            print(f"[Info] Not Found preprocessed_info.json:\n  {self.info_json_path}")
            sys.exit(1)

        num_classes = len(df_test["target"].unique())
        attr_name = info.get("attr_name", list(df_val.columns[:-1]))
        numerical_feature_count = info.get("numerical_feature_count", 0)
        numerical_feature_idx = info.get("numerical_feature_idx", [])
        categorical_feature_count = info.get("categorical_feature_count", 0)
        categorical_feature_idx = info.get("categorical_feature_idx", [])
        unique_values_per_categorical_feature = list(info.get("unique_values_per_categorical_feature", {}).values())

        for seed in sorted(seed_dfs.keys()):
            df_train = seed_dfs[seed]
            print(f"[Trainloader] seed{seed} has {len(df_train)} samples")
            train_attr = df_train.columns[:-1]
            X_train = df_train[train_attr].values
            y_train = df_train["target"].values
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            dst_train = TensorDataset(X_train_tensor, y_train_tensor)
            L = len(dst_train)
            real_bs = min(2048, L)
            trainloader = DataLoader(dst_train, batch_size=real_bs, shuffle=True, drop_last=True)
            trainloader_list.append(trainloader)

        val_attr = df_val.columns[:-1]
        X_val = df_val[val_attr].values
        y_val = df_val["target"].values
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        dst_val = TensorDataset(X_val_tensor, y_val_tensor)
        valloader = DataLoader(dst_val, batch_size=self.args.batch_train, shuffle=False)

        test_attr = df_test.columns[:-1]
        X_test = df_test[test_attr].values
        y_test = df_test["target"].values
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        dst_test = TensorDataset(X_test_tensor, y_test_tensor)
        testloader = DataLoader(dst_test, batch_size=self.args.batch_train, shuffle=False)

        return (
            trainloader_list,
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
