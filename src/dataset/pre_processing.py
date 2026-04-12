"""
80/10/10 train/val/test split, numeric scaling, and categorical encoding
(label_encoder / target_encoder / autoencoder).

Writes to dataset/preprocessed_datasets/<categorical_method>/<dataset>/
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import skew
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import resolve_categorical_method
from utils import set_seed


class DataPreprocessor:
    def __init__(
        self,
        dataset_name,
        dataset_dir=None,
        output_dir=None,
        model_root=None,
        categorical_method="autoencoder",
    ):
        self.dataset_name = dataset_name
        if dataset_dir is None:
            dataset_dir = os.path.join(_PROJECT_ROOT, "dataset", "download_datasets")
        if output_dir is None:
            output_dir = os.path.join(_PROJECT_ROOT, "dataset", "preprocessed_datasets")
        if model_root is None:
            model_root = os.path.join(_PROJECT_ROOT, "dataset", "categorical_encoder_cctc")

        self.dataset_dir = dataset_dir
        self.output_dir = os.path.join(output_dir, categorical_method.lower())
        self.model_root = model_root
        self.categorical_method = categorical_method.lower()
        self.data = None
        self.dataset_info = None
        self.df_tr = None
        self.df_va = None
        self.df_te = None
        self.duration = 0.0
        print(f"[Init] Processor for '{self.dataset_name}' created, method={self.categorical_method}")

    def load_dataset_info(self):
        path = os.path.join(self.dataset_dir, "datasets_info.json")
        print(f"[Load Info] Reading dataset info from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
        if self.dataset_name not in info:
            raise ValueError(
                f"Dataset {self.dataset_name} not in datasets_info.json. "
                f"Run: python dataset/download_dataset.py --datasets {self.dataset_name}"
            )
        self.dataset_info = info[self.dataset_name]
        print(f"[Load Info] Found dataset info keys: {list(self.dataset_info.keys())}")

    def load_data(self):
        path = os.path.join(self.dataset_dir, self.dataset_name, f"{self.dataset_name}.csv")
        print(f"[Load Data] Loading data from: {path}")
        self.data = pd.read_csv(path)
        print(f"[Load Data] Original data shape: {self.data.shape}")
        cols = self.dataset_info["attr_name"].copy()
        flags = self.dataset_info["cate_indicator"].copy()
        if "target" in self.data.columns and "target" not in cols:
            cols.append("target")
            flags.append(True)
        extras = set(self.data.columns) - set(cols)
        if extras:
            print(f"[Load Data] Dropping extra columns: {extras}")
            self.data.drop(columns=list(extras), inplace=True)
        self.data = self.data[cols]
        print(f"[Load Data] Filtered data shape: {self.data.shape}")
        self.dataset_info["attr_name"] = cols
        self.dataset_info["cate_indicator"] = flags
        attr_names = self.dataset_info["attr_name"]
        target_col = attr_names[-1]
        self.data[target_col] = pd.factorize(self.data[target_col])[0]

    def reorder_split(self):
        cols = self.dataset_info["attr_name"]
        flags = self.dataset_info["cate_indicator"]
        target = cols[-1]
        nums = [c for c, f in zip(cols[:-1], flags[:-1]) if not f]
        cats = [c for c, f in zip(cols[:-1], flags[:-1]) if f]
        order = nums + cats + [target]
        self.data = self.data[order]
        print(f"[Split] Reordered columns: nums={nums}, cats={cats}, target={target}")
        self.df_tr, tmp = train_test_split(
            self.data, test_size=0.2, random_state=42, stratify=self.data[target]
        )
        self.df_va, self.df_te = train_test_split(tmp, test_size=0.5, random_state=42, stratify=tmp[target])
        print(f"[Split] Train/Val/Test shapes: {self.df_tr.shape}, {self.df_va.shape}, {self.df_te.shape}")

    @staticmethod
    def detect_outliers_iqr(arr):
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        return np.any((arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr))

    @classmethod
    def choose_scaler(cls, df):
        scaler_map = {}
        for c in df.columns:
            arr = df[c].values
            if cls.detect_outliers_iqr(arr) or abs(skew(arr)) > 1:
                pipe = Pipeline(
                    [
                        ("primary", RobustScaler()),
                        ("secondary", MinMaxScaler(feature_range=(0, 1))),
                    ]
                )
            elif arr.max() - arr.min() > 100:
                pipe = Pipeline([("primary", MinMaxScaler(feature_range=(0, 1)))])
            else:
                pipe = Pipeline(
                    [
                        ("primary", StandardScaler()),
                        ("secondary", MinMaxScaler(feature_range=(0, 1))),
                    ]
                )
            scaler_map[c] = pipe
        print(f"[Numeric] Chosen scalers for columns: {list(scaler_map.keys())}")
        return scaler_map

    @staticmethod
    def apply_scaling(df, scaler_map):
        out = df.copy()
        for c, pipe in scaler_map.items():
            out[c] = pipe.transform(df[[c]]).ravel()
        return out

    def process_numerical(self):
        names = self.dataset_info["attr_name"][:-1]
        flags = self.dataset_info["cate_indicator"][:-1]
        num_cols = [c for c, f in zip(names, flags) if not f]
        if not num_cols:
            print("[Numeric] No numerical columns to process.")
            return
        print(f"[Numeric] Processing columns: {num_cols}")
        scaler_map = self.choose_scaler(self.df_tr[num_cols])
        for c, pipe in scaler_map.items():
            pipe.fit(self.df_tr[[c]])
        self.df_tr[num_cols] = self.apply_scaling(self.df_tr[num_cols], scaler_map)
        self.df_va[num_cols] = self.apply_scaling(self.df_va[num_cols], scaler_map)
        self.df_te[num_cols] = self.apply_scaling(self.df_te[num_cols], scaler_map)
        print("[Numeric] Scaling applied to train/val/test.")

    def process_categorical_le(self):
        start = time.time()
        feature_names = self.dataset_info["attr_name"][:-1]
        cate_flags = self.dataset_info["cate_indicator"][:-1]
        cat_cols = [name for name, flag in zip(feature_names, cate_flags) if flag]
        cat_maps = {}
        scalers = {}
        for col in cat_cols:
            col_train = self.df_tr[col].fillna("MISSING").astype(str)
            codes, uniques = pd.factorize(col_train)
            cat_maps[col] = {val: idx for idx, val in enumerate(uniques)}
            new_code = len(uniques)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(codes.reshape(-1, 1))
            scalers[col] = (scaler, new_code)
        for df in (self.df_tr, self.df_va, self.df_te):
            for col in cat_cols:
                vals = df[col].fillna("MISSING").astype(str)
                mapped = vals.map(cat_maps[col]).fillna(scalers[col][1]).astype(int)
                scaler = scalers[col][0]
                df[col] = scaler.transform(mapped.values.reshape(-1, 1)).flatten()
        self.duration = time.time() - start
        print(f"[Cat-LE] done in {self.duration:.3f}s; columns: {cat_cols}")

    def process_categorical_te(self, noise_level=0.01, n_splits=5):
        cat_maps = {}
        cat_defaults = {}
        start = time.time()
        feature_names = self.dataset_info["attr_name"][:-1]
        cate_flags = self.dataset_info["cate_indicator"][:-1]
        cat_cols = [n for n, f in zip(feature_names, cate_flags) if f]
        y_col = self.dataset_info["attr_name"][-1]
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for col in cat_cols:
            arr_tr = np.asarray(
                self.df_tr[col].fillna("MISSING").astype(str), dtype=object
            )
            y_tr = self.df_tr[y_col].values
            col_oof = np.zeros(len(arr_tr))
            for train_idx, val_idx in kf.split(arr_tr):
                df_fold = pd.DataFrame({"cat": arr_tr[train_idx], "y": y_tr[train_idx]})
                stats = df_fold.groupby("cat")["y"].agg(["count", "mean"])
                alpha = stats["count"].median()
                mu_all = y_tr.mean()
                stats["mu_smooth"] = (stats["count"] * stats["mean"] + alpha * mu_all) / (
                    stats["count"] + alpha
                )
                sigma = stats["mu_smooth"].std()
                noise = np.random.normal(0, noise_level * sigma, size=len(stats))
                stats["mu_noisy"] = stats["mu_smooth"] + noise
                mapping = stats["mu_noisy"].to_dict()
                cats_val = arr_tr[val_idx]
                col_oof[val_idx] = [mapping.get(c, mu_all) for c in cats_val]
            scaler = MinMaxScaler(feature_range=(0, 1))
            col_oof_scaled = scaler.fit_transform(col_oof.reshape(-1, 1)).flatten()
            self.df_tr[col] = col_oof_scaled
            df_full = pd.DataFrame({"cat": arr_tr, "y": y_tr})
            stats_full = df_full.groupby("cat")["y"].agg(["count", "mean"])
            alpha_full = stats_full["count"].median()
            mu_all = y_tr.mean()
            stats_full["mu_smooth"] = (stats_full["count"] * stats_full["mean"] + alpha_full * mu_all) / (
                stats_full["count"] + alpha_full
            )
            stats_full["te_scaled"] = scaler.transform(stats_full["mu_smooth"].values.reshape(-1, 1)).flatten()
            cat_maps[col] = stats_full["te_scaled"].to_dict()
            cat_defaults[col] = scaler.transform([[mu_all]]).item()
        for df_ in (self.df_va, self.df_te):
            for col in cat_cols:
                vals = df_[col].fillna("MISSING").astype(str)
                te = vals.map(cat_maps[col]).fillna(cat_defaults[col])
                df_[col] = te.values.reshape(-1, 1).flatten()
        self.duration = time.time() - start
        print(f"[Cat-TE KFold] {self.duration:.3f}s; columns: {cat_cols}")

    def process_categorical_autoencoder(self):
        names = self.dataset_info["attr_name"][:-1]
        flags = self.dataset_info["cate_indicator"][:-1]
        cat_cols = [col for col, is_cat in zip(names, flags) if is_cat]
        if not cat_cols:
            print("[Cat-AE] No categorical columns; skip SimilarityEncoder / AE.")
            self.duration = 0.0
            return
        try:
            from skrub import SimilarityEncoder
        except ImportError as e:
            raise ImportError(
                "categorical_method=autoencoder requires the 'skrub' package. "
                "Install with: pip install skrub"
            ) from e
        encoders = {}
        embeddings = []
        for col in cat_cols:
            print(f"[Cat-AE] Fitting SimilarityEncoder on '{col}'")
            self.df_tr[col] = self.df_tr[col].fillna("").astype(str)
            se = SimilarityEncoder(categories="auto")
            emb = se.fit_transform(np.asarray(self.df_tr[[col]], dtype=object))
            embeddings.append(emb)
            encoders[col] = se
        X_emb = np.hstack(embeddings)
        input_dim = X_emb.shape[1]
        latent_dim = len(cat_cols)
        hidden_dim = max(2, int(np.sqrt(input_dim * latent_dim)))
        print(f"[Cat-AE] input_dim={input_dim}, hidden_dim={hidden_dim}, latent_dim={latent_dim}")
        X_tensor = torch.tensor(X_emb, dtype=torch.float32)
        ds = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        model = Autoencoder(input_dim, hidden_dim, latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        start = time.time()
        model.train()
        for epoch in range(10):
            total = 0.0
            for batch_x, _ in loader:
                recon = model(batch_x)
                loss = criterion(recon, batch_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * batch_x.size(0)
            print(f"[Cat-AE] Epoch {epoch+1}: avg loss = {total/len(ds):.4f}")
        end = time.time()
        model.eval()
        with torch.no_grad():
            Z_tr = model.encoder(X_tensor).numpy()
        scaler = MinMaxScaler((0, 1))
        scaler.fit_transform(Z_tr)
        self.duration = end - start
        print(f"[Cat-AE] Categorical AE time: {self.duration:.3f}s")
        subdir = os.path.join(self.model_root, self.dataset_name, "cat_ae")
        os.makedirs(subdir, exist_ok=True)
        torch.save(model.encoder.state_dict(), os.path.join(subdir, "encoder.pt"))
        joblib.dump(encoders, os.path.join(subdir, "encoders.joblib"))
        joblib.dump(scaler, os.path.join(subdir, "scaler.joblib"))

        for split_df in (self.df_tr, self.df_va, self.df_te):
            mats = []
            for col in cat_cols:
                arr = np.asarray(
                    split_df[col].fillna("").astype(str), dtype=object
                ).reshape(-1, 1)
                mats.append(encoders[col].transform(arr))
            X_cat = np.hstack(mats)
            with torch.no_grad():
                Z = model.encoder(torch.tensor(X_cat, dtype=torch.float32)).numpy()
            Z_scaled = scaler.transform(Z)
            for j, col in enumerate(cat_cols):
                split_df[col] = Z_scaled[:, j]
            rename_map = {col: f"cat_ae_{j}" for j, col in enumerate(cat_cols)}
            split_df.rename(columns=rename_map, inplace=True)
        print("[Cat-AE] Categorical columns replaced by AE latent features.")

    def save_preprocessed_info(self):
        # Feature names must match CSV columns (e.g. cat_ae_* after autoencoder).
        target = self.df_tr.columns[-1]
        feat = list(self.df_tr.columns[:-1])
        class_count = int(self.df_tr[target].nunique())
        feature_count = len(feat)
        info = {
            "class_count": class_count,
            "feature_count": feature_count,
            "attr_name": feat,
            "numerical_feature_count": feature_count,
            "numerical_feature_idx": list(range(feature_count)),
            "categorical_feature_count": 0,
            "categorical_feature_idx": [],
            "unique_values_per_categorical_feature": {},
            "autoencoder_duration": self.duration,
        }
        out = os.path.join(self.output_dir, self.dataset_name)
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"{self.dataset_name}_preprocessed_info.json")
        with open(path, "w") as f:
            json.dump(info, f, indent=4)
        print(f"[Save Info] Preprocessed info saved to: {path}")

    def process_categorical(self):
        if self.categorical_method == "label_encoder":
            self.process_categorical_le()
        elif self.categorical_method == "autoencoder":
            self.process_categorical_autoencoder()
        elif self.categorical_method == "target_encoder":
            self.process_categorical_te()
        elif self.categorical_method == "na":
            print("No categorical encoding (na).")
        else:
            raise ValueError(f"Unknown categorical_method: {self.categorical_method}")

    def run_preprocessing(self):
        print(f"=== Starting preprocessing for {self.dataset_name} ===")
        self.load_dataset_info()
        self.load_data()
        self.reorder_split()
        self.process_numerical()
        self.process_categorical()
        out = os.path.join(self.output_dir, self.dataset_name)
        os.makedirs(out, exist_ok=True)
        self.df_tr.to_csv(os.path.join(out, f"{self.dataset_name}_train.csv"), index=False)
        self.df_va.to_csv(os.path.join(out, f"{self.dataset_name}_val.csv"), index=False)
        self.df_te.to_csv(os.path.join(out, f"{self.dataset_name}_test.csv"), index=False)
        print(f"[Save Data] Train/Val/Test CSVs saved to: {out}")
        self.save_preprocessed_info()
        print(f"=== Finished preprocessing for {self.dataset_name} ===\n")


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CSVs to dataset/preprocessed_datasets/<categorical_method>/<dataset>/ for CCTC and eval."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Adult",
        help="Key in datasets_info.json (must match download_dataset.py names).",
    )
    parser.add_argument(
        "--categorical_method",
        type=str,
        default=None,
        help="autoencoder | label_encoder | target_encoder | na. "
        "If omitted, uses dataset default from config (same as CCTC CLI).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cat_method = resolve_categorical_method(args.dataset, args.categorical_method)
    print(f"[pre_processing] Using categorical_method={cat_method} for dataset={args.dataset}")
    set_seed(args.seed)
    DataPreprocessor(args.dataset, categorical_method=cat_method).run_preprocessing()


if __name__ == "__main__":
    main()
