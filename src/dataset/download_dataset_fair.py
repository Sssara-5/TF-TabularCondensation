"""
Download Fair-CCTC datasets and update datasets_info.json.

Datasets:
  - ACSIncome          <- ACSIncome CA 2018 (categorical), sensitive=sex
  - ACSPublicCoverage  <- ACSPublicCoverage US 2018 (categorical), sensitive=race
  - credit             <- HyperGCL fair_data/credit, sensitive=age

Layout (same as download_dataset.py):
  dataset/download_datasets/<name>/<name>.csv
  dataset/download_datasets/datasets_info.json

Usage (from src/):
  python dataset/download_dataset_fair.py
  python dataset/download_dataset_fair.py --datasets ACSIncome,credit
  python dataset/download_dataset_fair.py --info-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from folktables import (
    ACSDataSource,
    ACSIncome,
    ACSPublicCoverage,
    BasicProblem,
    adult_filter,
    generate_categories,
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DEFAULT_DOWNLOAD_DIR = os.path.join(_PROJECT_ROOT, "dataset", "download_datasets")
DEFAULT_JSON_PATH = os.path.join(DEFAULT_DOWNLOAD_DIR, "datasets_info.json")
FOLKTABLES_ROOT = os.path.join(DEFAULT_DOWNLOAD_DIR, "_folktables")
CREDIT_CSV_URL = (
    "https://raw.githubusercontent.com/weitianxin/HyperGCL/"
    "master/data/fair_data/credit/credit.csv"
)

# ── ACS task definitions ─────────────────────────────────────────────────────

ACSIncome_SEX = BasicProblem(
    features=[
        "AGEP", "COW", "SCHL", "MAR",
        "OCCP", "POBP", "RELP", "WKHP",
        "SEX", "RAC1P",
    ],
    target="PINCP",
    target_transform=lambda x: x > 50000,
    group="SEX",
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)


def sex_transform(values):
    return (values == 2).astype(int)  # Female→1, Male→0


def race_transform(values):
    return (values != 1).astype(int)  # Non-White→1, White→0


# ── datasets_info helpers (from update_datasets_info_numeric.py) ──────────────

def _json_safe_class_distribution(series: pd.Series) -> Dict[str, int]:
    vc = series.value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.to_dict().items()}


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def _is_categorical_column(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().all() and len(numeric) == len(non_null):
        return False
    return True


def build_dataset_info(
    df: pd.DataFrame,
    target_col: str,
    sensitive_attribute: Optional[str],
    exclude_cols: List[str],
    force_all_numeric: bool = False,
) -> Dict[str, Any]:
    if target_col not in df.columns:
        raise ValueError(
            f"target_col={target_col!r} is not in the CSV columns. "
            f"Actual columns: {list(df.columns)}"
        )

    exclude = set(exclude_cols)
    exclude.add(target_col)

    feature_names = [c for c in df.columns if c not in exclude]
    if len(feature_names) == 0:
        raise ValueError(
            f"No features remain after excluding target and exclude_cols. "
            f"target_col={target_col!r}, exclude_cols={exclude_cols!r}"
        )

    sample_count = int(df.shape[0])
    feature_count = int(len(feature_names))
    y = df[target_col]
    class_distribution = _json_safe_class_distribution(y)
    class_count = int(len(class_distribution))

    if force_all_numeric:
        cate_indicator = [False] * feature_count
    else:
        cate_indicator = [_is_categorical_column(df[col]) for col in feature_names]

    numerical_feature_count = sum(1 for flag in cate_indicator if not flag)
    categorical_feature_count = sum(1 for flag in cate_indicator if flag)

    unique_values_per_categorical_feature = {
        col: int(df[col].nunique(dropna=True))
        for col, is_cat in zip(feature_names, cate_indicator)
        if is_cat
    }

    missing_values_per_feature = [int(df[col].isnull().sum()) for col in feature_names]
    total_missing_values = int(sum(missing_values_per_feature))
    denom = sample_count * feature_count
    missing_value_ratio = float(total_missing_values / denom) if denom else 0.0

    info: Dict[str, Any] = {
        "sample_count": sample_count,
        "class_count": class_count,
        "class_distribution": class_distribution,
        "feature_count": feature_count,
        "numerical_feature_count": numerical_feature_count,
        "categorical_feature_count": categorical_feature_count,
        "cate_indicator": cate_indicator,
        "attr_name": feature_names,
        "unique_values_per_categorical_feature": unique_values_per_categorical_feature,
        "missing_values_per_feature": missing_values_per_feature,
        "total_missing_values": total_missing_values,
        "missing_value_ratio": missing_value_ratio,
    }
    if sensitive_attribute is not None:
        info["sensitive_attribute"] = sensitive_attribute
    return info


def _print_fairness_stats(df: pd.DataFrame, g0_name: str, g1_name: str) -> None:
    mask0 = df["sensitive"] == 0
    mask1 = df["sensitive"] == 1
    n0, n1 = int(mask0.sum()), int(mask1.sum())
    pr0 = float(df.loc[mask0, "label"].mean()) if n0 else float("nan")
    pr1 = float(df.loc[mask1, "label"].mean()) if n1 else float("nan")
    ddp = abs(pr0 - pr1) if n0 and n1 else float("nan")
    print(f"  Columns:           {df.columns.tolist()}")
    print(f"  Total samples:     {len(df):,}")
    print(f"  Overall positive rate: {df['label'].mean():.4f}")
    print(f"  S=0 ({g0_name}): {n0:,}  ({n0 / len(df):.3f})  P(Y=1|S=0)={pr0:.4f}")
    print(f"  S=1 ({g1_name}): {n1:,}  ({n1 / len(df):.3f})  P(Y=1|S=1)={pr1:.4f}")
    print(f"  ΔDP:              {ddp:.4f}")


# ── downloaders ───────────────────────────────────────────────────────────────

def download_acs_categorical(
    task,
    states,
    out_path: str,
    group_transform,
    group_raw_col: str,
    g0_name: str,
    g1_name: str,
    data_source: ACSDataSource,
    definition_df: pd.DataFrame,
) -> pd.DataFrame:
    print(f"\n[Download] {out_path} ...")
    raw_data = data_source.get_data(states=states, download=True)
    categories = generate_categories(features=task.features, definition_df=definition_df)
    X_df, y_df, group_df = task.df_to_pandas(raw_data, categories=categories)

    df = X_df.copy()
    if group_raw_col in df.columns:
        df = df.drop(columns=[group_raw_col])

    df["sensitive"] = group_transform(group_df.values)
    df["label"] = y_df.values.astype(int)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[Saved] {out_path}")
    _print_fairness_stats(df, g0_name, g1_name)
    return df


def download_credit(out_path: str, force: bool = False) -> pd.DataFrame:
    """
    Download credit.csv from HyperGCL and rearrange columns so that
    NoDefaultNextMonth is the last column (target), matching fair_preprocessor.
    Sensitive attribute column: Age (already binary 0=old / 1=young).
    """
    print(f"\n[Download] credit from HyperGCL ...")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if os.path.exists(out_path) and not force:
        print(f"[Info] {out_path} already exists. Skipping download (use --force to overwrite).")
        return pd.read_csv(out_path)

    tmp_path = out_path + ".tmp"
    try:
        urlretrieve(CREDIT_CSV_URL, tmp_path)
        df = pd.read_csv(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # HyperGCL layout: NoDefaultNextMonth first → move to last.
    if "NoDefaultNextMonth" not in df.columns:
        raise ValueError(
            f"Unexpected credit.csv columns (missing NoDefaultNextMonth): {list(df.columns)}"
        )
    if "Age" not in df.columns:
        raise ValueError(f"Unexpected credit.csv columns (missing Age): {list(df.columns)}")

    feature_cols = [c for c in df.columns if c != "NoDefaultNextMonth"]
    df = df[feature_cols + ["NoDefaultNextMonth"]]
    df.to_csv(out_path, index=False)

    n = len(df)
    age0 = int((df["Age"] == 0).sum())
    age1 = int((df["Age"] == 1).sum())
    y = df["NoDefaultNextMonth"]
    print(f"[Saved] {out_path}")
    print(f"  Columns:           {df.columns.tolist()}")
    print(f"  Total samples:     {n:,}")
    print(f"  Overall positive rate: {y.mean():.4f}")
    print(f"  S=0 (old/Age=0): {age0:,}  ({age0 / n:.3f})")
    print(f"  S=1 (young/Age=1): {age1:,}  ({age1 / n:.3f})")
    return df


# ── dataset registry ──────────────────────────────────────────────────────────

# name -> metadata used for download + datasets_info.json
DATASET_SPECS = {
    "ACSIncome": {
        "kind": "acs",
        "task": ACSIncome_SEX,
        "states": ["CA"],
        "group_transform": sex_transform,
        "group_raw_col": "SEX",
        "g0_name": "male",
        "g1_name": "female",
        "target_col": "label",
        "sensitive_attribute": "sex",
        "exclude_cols": ["sensitive"],
    },
    "ACSPublicCoverage": {
        "kind": "acs",
        "task": ACSPublicCoverage,
        "states": None,  # all US states
        "group_transform": race_transform,
        "group_raw_col": "RAC1P",
        "g0_name": "white",
        "g1_name": "non-white",
        "target_col": "label",
        "sensitive_attribute": "race",
        "exclude_cols": ["sensitive"],
    },
    "credit": {
        "kind": "credit",
        "target_col": "NoDefaultNextMonth",
        "sensitive_attribute": "age",
        "exclude_cols": [],
        "force_all_numeric": True,
    },
}


class DownloadFairDataset:
    def __init__(self, download_dir: Optional[str] = None, json_path: Optional[str] = None):
        self.download_dir = download_dir or DEFAULT_DOWNLOAD_DIR
        self.json_path = json_path or os.path.join(self.download_dir, "datasets_info.json")
        os.makedirs(self.download_dir, exist_ok=True)
        self._data_source: Optional[ACSDataSource] = None
        self._definition_df: Optional[pd.DataFrame] = None

    def _acs_source(self) -> tuple[ACSDataSource, pd.DataFrame]:
        if self._data_source is None:
            os.makedirs(FOLKTABLES_ROOT, exist_ok=True)
            self._data_source = ACSDataSource(
                survey_year="2018",
                horizon="1-Year",
                survey="person",
                root_dir=FOLKTABLES_ROOT,
            )
            self._definition_df = self._data_source.get_definitions(download=True)
        return self._data_source, self._definition_df

    def _csv_path(self, name: str) -> str:
        # Same layout as download_dataset.py: download_datasets/<name>/<name>.csv
        return os.path.join(self.download_dir, name, f"{name}.csv")

    def _iter_names(self, names: Optional[List[str]]):
        if not names:
            return list(DATASET_SPECS.keys())
        missing = [n for n in names if n not in DATASET_SPECS]
        if missing:
            raise ValueError(f"Unknown dataset(s): {missing}. Valid: {list(DATASET_SPECS)}")
        return names

    def download(self, names: Optional[List[str]] = None, force: bool = False) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        for name in self._iter_names(names):
            spec = DATASET_SPECS[name]
            out_path = self._csv_path(name)

            if spec["kind"] == "acs":
                if os.path.exists(out_path) and not force:
                    print(f"[Info] {out_path} already exists. Skipping download (use --force to overwrite).")
                    results[name] = pd.read_csv(out_path)
                    continue
                data_source, definition_df = self._acs_source()
                results[name] = download_acs_categorical(
                    task=spec["task"],
                    states=spec["states"],
                    out_path=out_path,
                    group_transform=spec["group_transform"],
                    group_raw_col=spec["group_raw_col"],
                    g0_name=spec["g0_name"],
                    g1_name=spec["g1_name"],
                    data_source=data_source,
                    definition_df=definition_df,
                )
            elif spec["kind"] == "credit":
                results[name] = download_credit(out_path, force=force)
            else:
                raise ValueError(f"Unknown kind: {spec['kind']}")
        return results

    def update_info(self, names: Optional[List[str]] = None) -> Dict[str, Any]:
        all_info = _load_json(self.json_path)
        for name in self._iter_names(names):
            spec = DATASET_SPECS[name]
            csv_path = self._csv_path(name)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found for {name}: {csv_path}. Run download first.")

            df = pd.read_csv(csv_path)
            info = build_dataset_info(
                df=df,
                target_col=spec["target_col"],
                sensitive_attribute=spec["sensitive_attribute"],
                exclude_cols=list(spec.get("exclude_cols") or []),
                force_all_numeric=bool(spec.get("force_all_numeric", False)),
            )
            all_info[name] = info

            cat_cols = [n for n, f in zip(info["attr_name"], info["cate_indicator"]) if f]
            num_cols = [n for n, f in zip(info["attr_name"], info["cate_indicator"]) if not f]
            print(f"\n[Info] Updated datasets_info entry: {name}")
            print(f"  sample_count={info['sample_count']}, feature_count={info['feature_count']}")
            print(f"  numerical={info['numerical_feature_count']}, categorical={info['categorical_feature_count']}")
            print(f"  sensitive_attribute={info.get('sensitive_attribute')}")
            print(f"  numerical cols: {num_cols}")
            print(f"  categorical cols: {cat_cols}")

        _save_json(self.json_path, all_info)
        print(f"\n[OK] Written/updated: {self.json_path}")
        return all_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download Fair-CCTC datasets (ACSIncome, ACSPublicCoverage, credit) "
            "and merge statistics into datasets_info.json."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated names. Empty = all (ACSIncome,ACSPublicCoverage,credit).",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=DEFAULT_DOWNLOAD_DIR,
        help="Directory for CSV outputs and datasets_info.json",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="",
        help="datasets_info.json path (default: <download-dir>/datasets_info.json)",
    )
    parser.add_argument("--download-only", action="store_true", help="Only download CSVs")
    parser.add_argument("--info-only", action="store_true", help="Only refresh datasets_info.json from existing CSVs")
    parser.add_argument("--force", action="store_true", help="Re-download even if CSV already exists")
    args = parser.parse_args()

    names = [x.strip() for x in args.datasets.split(",") if x.strip()] or None
    json_path = args.json_path or None
    dl = DownloadFairDataset(download_dir=args.download_dir, json_path=json_path)

    if args.info_only:
        dl.update_info(names=names)
        return

    results = dl.download(names=names, force=args.force)
    if not args.download_only:
        dl.update_info(names=names)

    print("\n" + "=" * 52)
    print("All done!")
    print("=" * 52)
    for name, df in results.items():
        print(f"  {name:<24} {len(df):>12,} rows  {df.shape[1]} columns")


if __name__ == "__main__":
    main()
