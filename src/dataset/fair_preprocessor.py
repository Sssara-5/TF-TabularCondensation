import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy.stats import skew
from skrub import SimilarityEncoder

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from utils import set_seed
except Exception:
    def set_seed(seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)


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


class FairDataPreprocessor:
    """
    Fair-CCTC preprocessing.

    Column order after preprocessing:
        sensitive + label-encoding group + target-encoding group
        + autoencoding group + target

    Rules:
        1. The last column is always renamed to 'target'.
        2. Sensitive semantic type (sex/race/age) is read from datasets_info.json
           field 'sensitive_attribute' (optional constructor override).
        3. Sensitive column is always named 'sensitive' (first column). If CSV has no
           'sensitive' column, locate the source column by semantic type / name, encode,
           rename to 'sensitive', and move to the first column.
        4. Sensitive attribute is converted to binary 0/1 (see _standardize_sensitive_column).
        5. Non-sensitive feature grouping (in infer_feature_groups):
            - numeric with <= 2 unique values: label encoding group
            - numeric with > 2 unique values: target encoding group
            - non-numeric/text: autoencoding group
        6. Each group is processed separately.
    """

    SENSITIVE_COL = "sensitive"
    SEMANTIC_SENSITIVE_NAMES = frozenset({"sex", "gender", "race", "age"})

    def __init__(
        self,
        dataset_name,
        sensitive_attr=None,
        dataset_dir=None,
        output_dir=None,
        model_root=None,
        test_size=0.2,
        val_size=0.5,
        random_state=42,
    ):
        self.dataset_name = dataset_name
        self._sensitive_attr_override = sensitive_attr
        self.sensitive_attr = None  # semantic type: sex / race / age (from datasets_info.json)
        self.sensitive_col = self.SENSITIVE_COL  # unified column name in processed data
        self.dataset_info = None

        # Same layout as pre_processing.py / download_dataset.py:
        #   download_datasets/<name>/<name>.csv
        #   preprocessed_datasets_fair/<name>/
        if dataset_dir is None:
            dataset_dir = os.path.join(_PROJECT_ROOT, "dataset", "download_datasets")
        if output_dir is None:
            output_dir = os.path.join(_PROJECT_ROOT, "dataset", "preprocessed_datasets_fair")
        if model_root is None:
            model_root = os.path.join(
                _PROJECT_ROOT, "dataset", "categorical_encoder_autoencoder_fair"
            )

        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.model_root = model_root
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.data = None
        self.df_tr = None
        self.df_va = None
        self.df_te = None

        self.label_cols = []
        self.target_encoding_cols = []
        self.autoencoding_cols = []
        self.feature_order = []
        self.duration = 0.0

        print(f"[Init] Fair processor for '{dataset_name}'")

    def load_datasets_info(self):
        """Load sensitive_attribute (sex/race/age) from datasets_info.json."""
        info_path = os.path.join(self.dataset_dir, "datasets_info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"datasets_info.json not found: {info_path}")

        with open(info_path, "r", encoding="utf-8") as f:
            all_info = json.load(f)

        info_key = self.dataset_name
        if info_key not in all_info:
            by_lower = {k.lower(): k for k in all_info}
            info_key = by_lower.get(self.dataset_name.lower())
        if info_key is None:
            raise ValueError(
                f"Dataset '{self.dataset_name}' not in {info_path}. "
                f"Available: {list(all_info.keys())}"
            )
        self.dataset_info_key = info_key
        self.dataset_info = all_info[info_key]
        if "sensitive_attribute" not in self.dataset_info:
            raise ValueError(
                f"Missing 'sensitive_attribute' for '{self.dataset_name}' in datasets_info.json"
            )

        json_attr = self.dataset_info["sensitive_attribute"]
        if self._sensitive_attr_override is not None:
            print(
                f"[Datasets Info] sensitive_attribute override: "
                f"{self._sensitive_attr_override} (JSON: {json_attr})"
            )
            self.sensitive_attr = self._sensitive_attr_override
        else:
            self.sensitive_attr = json_attr

        print(
            f"[Datasets Info] '{self.dataset_name}' -> sensitive_attribute='{self.sensitive_attr}'"
        )

    @staticmethod
    def _to_2d(arr):
        return np.asarray(arr).reshape(-1, 1)

    @staticmethod
    def _categorical_str_2d(series, missing="MISSING"):
        return np.asarray(series.fillna(missing).astype(str), dtype=object).reshape(-1, 1)

    @staticmethod
    def _normalize_name(name):
        return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")

    def _find_column(self, requested_name):
        requested = self._normalize_name(requested_name)
        for col in self.data.columns:
            if self._normalize_name(col) == requested:
                return col
        raise ValueError(
            f"Sensitive attribute '{requested_name}' not found. "
            f"Available columns: {list(self.data.columns)}"
        )

    def _column_exists(self, requested_name):
        requested = self._normalize_name(requested_name)
        return any(self._normalize_name(col) == requested for col in self.data.columns)

    @staticmethod
    def _is_zero_one_values(unique_values):
        if len(unique_values) != 2:
            return False
        try:
            return {float(v) for v in unique_values} == {0.0, 1.0}
        except (TypeError, ValueError):
            return False

    def _binary_sensitive_to_01(self, series):
        """
        If exactly two unique values:
          - already {0, 1} -> keep unchanged
          - otherwise -> map smaller/lex-first to 0, other to 1

        Returns:
            (encoded_series, mode) with mode in {'unchanged', 'remapped'}, or (None, None)
            when not exactly two unique values.
        """
        s = series.copy()
        non_null = s.dropna()
        if non_null.empty:
            return pd.Series(0, index=s.index, dtype=int), "unchanged"

        numeric = pd.to_numeric(non_null, errors="coerce")
        all_numeric = numeric.notna().all() and len(numeric) == len(non_null)

        if all_numeric:
            unique = sorted(numeric.unique().tolist())
            if len(unique) != 2:
                return None, None
            out = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
            if self._is_zero_one_values(unique):
                return out, "unchanged"
            mapping = {unique[0]: 0, unique[1]: 1}
            return pd.to_numeric(s, errors="coerce").map(mapping).fillna(0).astype(int), "remapped"

        unique = sorted(non_null.unique().tolist(), key=lambda x: str(x))
        if len(unique) != 2:
            return None, None
        if self._is_zero_one_values(unique):
            return pd.to_numeric(s, errors="coerce").fillna(0).astype(int), "unchanged"
        mapping = {unique[0]: 0, unique[1]: 1}
        return s.map(mapping).fillna(0).astype(int), "remapped"

    def _resolve_sensitive_source_column(self):
        """Pick CSV column to read before renaming to 'sensitive'."""
        if self._column_exists(self.SENSITIVE_COL):
            return self._find_column(self.SENSITIVE_COL)
        return self._find_column(self.sensitive_attr)

    def _standardize_sensitive_column(self):
        """
        Encode sensitive attribute (from datasets_info.json) and place as first column 'sensitive'.

        Decision order:
          1) Exactly 2 unique values and already {0, 1} -> keep as-is
          2) semantic type in {sex, race, age} from JSON -> encode_sensitive_attribute()
          3) Exactly 2 unique values but not {0, 1} -> map smaller/lex-first -> 0, other -> 1
          4) Otherwise -> encode_sensitive_attribute() or error
        """
        sens_col = self._resolve_sensitive_source_column()
        series = self.data[sens_col]
        encoded, mode = self._binary_sensitive_to_01(series)
        use_semantic = self._normalize_name(self.sensitive_attr) in self.SEMANTIC_SENSITIVE_NAMES

        if mode == "unchanged":
            print(f"[Load Data] Sensitive '{sens_col}' already binary 0/1, kept unchanged.")
        elif use_semantic:
            encoded = self.encode_sensitive_attribute(series, self.sensitive_attr)
            print(
                f"[Load Data] Sensitive '{sens_col}' encoded by semantic rule "
                f"({self.sensitive_attr} from datasets_info.json)."
            )
        elif encoded is None:
            encoded = self.encode_sensitive_attribute(series, self.sensitive_attr)
            print(
                f"[Load Data] Sensitive '{sens_col}' encoded by semantic rule "
                f"({self.sensitive_attr})."
            )
        else:
            print(f"[Load Data] Sensitive '{sens_col}' has 2 values (not 0/1), mapped to 0/1.")

        if self._normalize_name(sens_col) != self._normalize_name(self.SENSITIVE_COL):
            self.data = self.data.drop(columns=[sens_col])
        self.data[self.SENSITIVE_COL] = np.asarray(encoded, dtype=int)

        feat_cols = [c for c in self.data.columns if c not in (self.SENSITIVE_COL, "target")]
        self.data = self.data[[self.SENSITIVE_COL] + feat_cols + ["target"]]
        print(
            f"[Load Data] Sensitive column standardized as '{self.SENSITIVE_COL}' "
            f"(semantic: {self.sensitive_attr}), placed first."
        )

    def load_data(self):
        self.load_datasets_info()

        path = os.path.join(
            self.dataset_dir, self.dataset_name, f"{self.dataset_name}.csv"
        )
        print(f"[Load Data] Loading data from: {path}")
        self.data = pd.read_csv(path)

        if self.data.shape[1] < 2:
            raise ValueError("Dataset must contain at least one feature column and one target column.")

        # Last column is always target.
        old_target = self.data.columns[-1]
        self.data = self.data.rename(columns={old_target: "target"})
        print(f"[Load Data] Last column '{old_target}' renamed to 'target'.")

        # Convert target to 0..C-1.
        self.data["target"] = pd.factorize(self.data["target"])[0]

        # Encode sensitive attribute; rename to 'sensitive' and move to first column.
        self._standardize_sensitive_column()

        print(f"[Load Data] Data shape: {self.data.shape}")

    def encode_sensitive_attribute(self, series, sensitive_attr):
        """Semantic encoding for sex / race / age when not handled by _binary_sensitive_to_01."""
        name = self._normalize_name(sensitive_attr)
        s = series.copy()

        if name == "sex" or name == "gender":
            text = s.fillna("MISSING").astype(str).str.strip().str.lower()
            # 0 = Male, 1 = Female. Unknown values are mapped to 1 to avoid treating them as Male.
            return np.where(text.isin(["male", "m", "man", "men", "0"]), 0, 1).astype(int)

        if name == "race":
            text = s.fillna("MISSING").astype(str).str.strip().str.lower()
            # 0 = White, 1 = Non-White.
            return np.where(text.eq("white"), 0, 1).astype(int)

        if name == "age":
            age = pd.to_numeric(s, errors="coerce")
            if age.isna().any():
                raise ValueError("Age sensitive attribute contains non-numeric values.")
            # 0 = old, age >= 25. 1 = young, age < 25.
            return np.where(age >= 25, 0, 1).astype(int)

        raise ValueError(
            "Unsupported sensitive attribute. Supported names: sex/gender, Race, Age. "
            f"Got: {sensitive_attr}"
        )

    def infer_feature_groups(self):
        self.label_cols = []
        self.target_encoding_cols = []
        self.autoencoding_cols = []

        for col in self.data.columns:
            if col in [self.sensitive_col, "target"]:
                continue

            numeric = pd.to_numeric(self.data[col], errors="coerce")
            is_numeric = numeric.notna().all()

            if is_numeric:
                # Store as numeric after successful detection.
                self.data[col] = numeric
                nunique = self.data[col].nunique(dropna=True)
                if nunique <= 2:
                    self.label_cols.append(col)
                else:
                    self.target_encoding_cols.append(col)
            else:
                self.data[col] = self.data[col].astype(object)
                self.autoencoding_cols.append(col)

        self.feature_order = (
            [self.sensitive_col]
            + self.label_cols
            + self.target_encoding_cols
            + self.autoencoding_cols
            + ["target"]
        )
        self.data = self.data[self.feature_order]

        print("[Infer Groups]")
        print(f"  sensitive: {self.sensitive_col} (semantic: {self.sensitive_attr})")
        print(f"  label encoding group: {self.label_cols}")
        print(f"  target encoding group: {self.target_encoding_cols}")
        print(f"  autoencoding group: {self.autoencoding_cols}")
        print(f"  final order: {self.feature_order}")

    def split_data(self):
        self.df_tr, tmp = train_test_split(
            self.data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.data["target"],
        )
        self.df_va, self.df_te = train_test_split(
            tmp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=tmp["target"],
        )
        self.df_tr = self.df_tr.reset_index(drop=True)
        self.df_va = self.df_va.reset_index(drop=True)
        self.df_te = self.df_te.reset_index(drop=True)
        print(f"[Split] Train/Val/Test: {self.df_tr.shape}, {self.df_va.shape}, {self.df_te.shape}")

    def process_label_encoding_group(self):
        if not self.label_cols:
            print("[Label Encoding] No columns.")
            return

        start = time.time()
        for col in self.label_cols:
            # For binary numeric columns, this maps the smaller value to 0 and larger value to 1.
            values = sorted(self.df_tr[col].dropna().unique().tolist())
            if len(values) == 1:
                mapping = {values[0]: 0}
                default = 0
            else:
                mapping = {values[0]: 0, values[1]: 1}
                default = 0

            for df in (self.df_tr, self.df_va, self.df_te):
                df[col] = df[col].map(mapping).fillna(default).astype(float)

        self.duration += time.time() - start
        print(f"[Label Encoding] Processed: {self.label_cols}")

    def process_target_encoding_group(self, noise_level=0.01, n_splits=5):
        if not self.target_encoding_cols:
            print("[Target Encoding] No columns.")
            return

        start = time.time()
        self.te_maps = {}
        self.te_defaults = {}
        self.te_scalers = {}

        y_col = "target"
        y_tr = self.df_tr[y_col].values
        mu_all = float(np.mean(y_tr))
        n_splits = min(n_splits, len(self.df_tr))

        for col in self.target_encoding_cols:
            # Numeric columns with many values are treated by exact-value target encoding.
            # This keeps one output column per original column.
            arr_tr = self.df_tr[col].fillna("MISSING").astype(str).values
            col_oof = np.zeros(len(arr_tr), dtype=float)

            if n_splits >= 2:
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                for train_idx, val_idx in kf.split(arr_tr):
                    df_fold = pd.DataFrame({"value": arr_tr[train_idx], "y": y_tr[train_idx]})
                    stats = df_fold.groupby("value")["y"].agg(["count", "mean"])
                    alpha = max(float(stats["count"].median()), 1.0)
                    stats["smooth"] = (stats["count"] * stats["mean"] + alpha * mu_all) / (
                        stats["count"] + alpha
                    )
                    sigma = float(stats["smooth"].std()) if len(stats) > 1 else 0.0
                    if sigma > 0 and noise_level > 0:
                        stats["smooth"] += np.random.normal(0, noise_level * sigma, size=len(stats))
                    mapping = stats["smooth"].to_dict()
                    col_oof[val_idx] = [mapping.get(v, mu_all) for v in arr_tr[val_idx]]
            else:
                col_oof[:] = mu_all

            scaler = MinMaxScaler(feature_range=(0, 1))
            self.df_tr[col] = scaler.fit_transform(col_oof.reshape(-1, 1)).flatten()

            df_full = pd.DataFrame({"value": arr_tr, "y": y_tr})
            stats_full = df_full.groupby("value")["y"].agg(["count", "mean"])
            alpha_full = max(float(stats_full["count"].median()), 1.0)
            stats_full["smooth"] = (
                stats_full["count"] * stats_full["mean"] + alpha_full * mu_all
            ) / (stats_full["count"] + alpha_full)
            stats_full["scaled"] = scaler.transform(
                stats_full["smooth"].values.reshape(-1, 1)
            ).flatten()

            self.te_maps[col] = stats_full["scaled"].to_dict()
            self.te_defaults[col] = float(scaler.transform([[mu_all]]).item())
            self.te_scalers[col] = scaler

            for df in (self.df_va, self.df_te):
                vals = df[col].fillna("MISSING").astype(str)
                df[col] = vals.map(self.te_maps[col]).fillna(self.te_defaults[col]).astype(float)

        self.duration += time.time() - start
        print(f"[Target Encoding] Processed: {self.target_encoding_cols}")

    def process_autoencoding_group(self, epochs=10, batch_size=256, lr=1e-3):
        if not self.autoencoding_cols:
            print("[Autoencoding] No columns.")
            return

        start = time.time()
        encoders = {}
        embeddings = []

        for col in self.autoencoding_cols:
            print(f"[Autoencoding] Fitting SimilarityEncoder on '{col}'")
            se = SimilarityEncoder(categories="auto")
            emb = se.fit_transform(self._categorical_str_2d(self.df_tr[col]))
            embeddings.append(emb)
            encoders[col] = se

        X_emb = np.hstack(embeddings).astype(np.float32)
        input_dim = X_emb.shape[1]
        latent_dim = len(self.autoencoding_cols)
        hidden_dim = max(2, int(np.sqrt(input_dim * latent_dim)))

        print(
            f"[Autoencoding] input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"latent_dim={latent_dim}"
        )

        X_tensor = torch.tensor(X_emb, dtype=torch.float32)
        ds = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        model = Autoencoder(input_dim, hidden_dim, latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            total = 0.0
            for batch_x, _ in loader:
                recon = model(batch_x)
                loss = criterion(recon, batch_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * batch_x.size(0)
            print(f"[Autoencoding] Epoch {epoch + 1}: avg loss = {total / len(ds):.4f}")

        model.eval()
        with torch.no_grad():
            Z_tr = model.encoder(X_tensor).numpy()

        scaler = MinMaxScaler((0, 1))
        scaler.fit(Z_tr)

        subdir = os.path.join(self.model_root, self.dataset_name, "cat_ae")
        os.makedirs(subdir, exist_ok=True)
        torch.save(model.encoder.state_dict(), os.path.join(subdir, "encoder.pt"))
        joblib.dump(encoders, os.path.join(subdir, "similarity_encoders.joblib"))
        joblib.dump(scaler, os.path.join(subdir, "latent_scaler.joblib"))

        for split_df in (self.df_tr, self.df_va, self.df_te):
            mats = []
            for col in self.autoencoding_cols:
                mats.append(encoders[col].transform(self._categorical_str_2d(split_df[col])))
            X_cat = np.hstack(mats).astype(np.float32)
            with torch.no_grad():
                Z = model.encoder(torch.tensor(X_cat, dtype=torch.float32)).numpy()
            Z_scaled = scaler.transform(Z)
            for j, col in enumerate(self.autoencoding_cols):
                split_df[col] = Z_scaled[:, j]

        self.duration += time.time() - start
        print(f"[Autoencoding] Processed: {self.autoencoding_cols}")

    def save_preprocessed_info(self):
        feat = self.feature_order[:-1]
        info = {
            "class_count": int(self.df_tr["target"].nunique()),
            "feature_count": len(feat),
            "attr_name": feat,
            "target_name": "target",
            "sensitive_col": self.sensitive_col,
            "sensitive_attr": self.sensitive_attr,
            "sensitive_idx": 0,
            "label_encoding_cols": self.label_cols,
            "target_encoding_cols": self.target_encoding_cols,
            "autoencoding_cols": self.autoencoding_cols,
            "feature_order": self.feature_order,
            "numerical_feature_count": len(feat),
            "numerical_feature_idx": list(range(len(feat))),
            "categorical_feature_count": 0,
            "categorical_feature_idx": [],
            "preprocessing_duration": self.duration,
        }

        out = os.path.join(self.output_dir, self.dataset_name)
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"{self.dataset_name}_preprocessed_info.json")
        with open(path, "w") as f:
            json.dump(info, f, indent=4)
        print(f"[Save Info] Saved: {path}")

    def save_data(self):
        out = os.path.join(self.output_dir, self.dataset_name)
        os.makedirs(out, exist_ok=True)
        self.df_tr.to_csv(os.path.join(out, f"{self.dataset_name}_train.csv"), index=False)
        self.df_va.to_csv(os.path.join(out, f"{self.dataset_name}_val.csv"), index=False)
        self.df_te.to_csv(os.path.join(out, f"{self.dataset_name}_test.csv"), index=False)
        print(f"[Save Data] Train/Val/Test saved to: {out}")

    def run_preprocessing(self):
        print(f"=== Starting fair preprocessing for {self.dataset_name} ===")
        self.load_data()
        self.infer_feature_groups()
        self.split_data()
        self.process_label_encoding_group()
        self.process_target_encoding_group()
        self.process_autoencoding_group()

        # Reorder again after all transforms to guarantee final layout.
        for name in ("df_tr", "df_va", "df_te"):
            df = getattr(self, name)
            setattr(self, name, df[self.feature_order])

        self.save_data()
        self.save_preprocessed_info()
        print(f"=== Finished fair preprocessing for {self.dataset_name} ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fair-CCTC preprocessing.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="credit",
        help="Dataset key in download_datasets/datasets_info.json (e.g. credit, ACSIncome).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    FairDataPreprocessor(args.dataset).run_preprocessing()
