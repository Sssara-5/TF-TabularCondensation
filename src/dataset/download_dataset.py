"""Download raw tabular CSVs from OpenML and build datasets_info.json for pre_processing."""
import argparse
import json
import os
import sys

import numpy as np
import openml
import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class DownloadDataset:
    DATASET_MAP = {
        "Adult": [45068, "adult", 4],
        "electricity": [43953, "electricity", 6],
        "Covertype": [1596, "Covertype", 4],
        "airlines": [41672, "airlines", 2],
        "Epsilon": [45575, "Epsilon", 1],
        "road-safety": [44161, "road-safety", 6],
        "Diabetes130US": [4541, "Diabetes130US", 1],
        "Jannis": [41168, "jannis", 1],
        "Higgs": [44092, "Higgs", 6],
        "Microsoft": [45579, "Microsoft", 2],
    }

    def __init__(self, download_path=None):
        if download_path is None:
            download_path = os.path.join(_PROJECT_ROOT, "dataset", "download_datasets")
        self.download_path = download_path
        os.makedirs(self.download_path, exist_ok=True)
        self.datasets_info = {}

    def _iter_map(self, names):
        if not names:
            return self.DATASET_MAP.items()
        missing = [n for n in names if n not in self.DATASET_MAP]
        if missing:
            raise ValueError(f"Unknown dataset(s): {missing}. Valid: {list(self.DATASET_MAP)}")
        return [(n, self.DATASET_MAP[n]) for n in names]

    def download_and_save(self, names=None):
        for dataset_name, info in tqdm(list(self._iter_map(names)), desc="Downloading datasets"):
            dataset_id = info[0]
            dataset_folder = os.path.join(self.download_path, dataset_name)
            dataset_file = os.path.join(dataset_folder, f"{dataset_name}.csv")
            os.makedirs(dataset_folder, exist_ok=True)

            if os.path.exists(dataset_file):
                print(f"[Info] {dataset_name}.csv already exists. Skipping download.")
                continue

            try:
                print(f"[Downloading] Dataset: {dataset_name} (ID: {dataset_id})")
                openml_dataset = openml.datasets.get_dataset(dataset_id)
                X, y, _, attr_name = openml_dataset.get_data(target=openml_dataset.default_target_attribute)
                if y is not None:
                    X["target"] = y
                X.to_csv(dataset_file, index=False, encoding="utf-8")
                print(f"[Saved] {dataset_name}.csv saved to {dataset_file}")
            except Exception as e:
                print(f"[Error] Failed to download {dataset_name} (ID: {dataset_id}): {e}")

    def extract_info(self, names=None):
        for dataset_name, info in tqdm(list(self._iter_map(names)), desc="Extracting dataset info"):
            dataset_id = info[0]
            print(f"[Info] Processing dataset: {dataset_name} (ID: {dataset_id})")
            try:
                openml_dataset = openml.datasets.get_dataset(dataset_id)
                target_attribute = openml_dataset.default_target_attribute or None
                if target_attribute is None:
                    print(f"[Warning] Dataset {dataset_name} has no default target attribute.")
                X, y, cate_indicator, attr_name = openml_dataset.get_data(target=target_attribute)
                cate_indicator = cate_indicator.tolist() if hasattr(cate_indicator, "tolist") else cate_indicator
                sample_count = X.shape[0] if hasattr(X, "shape") else len(X)
                feature_count = len(attr_name)

                if y is not None:
                    unique_classes, class_counts = np.unique(y, return_counts=True)
                    class_distribution = {
                        str(unique_classes[i]): int(class_counts[i]) for i in range(len(unique_classes))
                    }
                    class_count = len(unique_classes)
                else:
                    class_distribution = None
                    class_count = 0

                numerical_feature_count = sum(1 for val in cate_indicator if not val)
                categorical_feature_count = sum(1 for val in cate_indicator if val)
                unique_values_per_categorical_feature = {
                    attr_name[i]: len(np.unique(X[attr_name[i]]))
                    for i in range(len(attr_name))
                    if cate_indicator[i]
                }
                if isinstance(X, pd.DataFrame):
                    missing_values_per_feature = X.isnull().sum().tolist()
                else:
                    missing_values_per_feature = [0] * len(attr_name)
                total_missing_values = sum(missing_values_per_feature)
                missing_value_ratio = (
                    total_missing_values / (sample_count * feature_count) if feature_count > 0 else 0
                )

                self.datasets_info[dataset_name] = {
                    "sample_count": sample_count,
                    "class_count": class_count,
                    "class_distribution": class_distribution,
                    "feature_count": feature_count,
                    "numerical_feature_count": numerical_feature_count,
                    "categorical_feature_count": categorical_feature_count,
                    "cate_indicator": cate_indicator,
                    "attr_name": attr_name,
                    "unique_values_per_categorical_feature": unique_values_per_categorical_feature,
                    "missing_values_per_feature": missing_values_per_feature,
                    "total_missing_values": total_missing_values,
                    "missing_value_ratio": missing_value_ratio,
                }
            except Exception as e:
                print(f"[Error] Dataset {dataset_name} (ID: {dataset_id}) failed: {type(e).__name__}: {e}")

        json_path = os.path.join(self.download_path, "datasets_info.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                old_info = json.load(f)
        else:
            old_info = {}
        merged_info = {**old_info, **self.datasets_info}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(merged_info, f, ensure_ascii=False, indent=4)
        print(f"[Info] All dataset information saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download OpenML datasets under <project>/dataset/download_datasets/"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated names (e.g. Covertype,Adult). Empty = all in DATASET_MAP.",
    )
    parser.add_argument("--download-only", action="store_true", help="Only download CSVs, no datasets_info.json")
    parser.add_argument("--info-only", action="store_true", help="Only refresh datasets_info.json from OpenML (no CSV write)")
    args = parser.parse_args()
    names = [x.strip() for x in args.datasets.split(",") if x.strip()] or None

    dl = DownloadDataset()
    if args.info_only:
        dl.extract_info(names=names)
        return
    dl.download_and_save(names=names)
    if not args.download_only:
        dl.extract_info(names=names)


if __name__ == "__main__":
    main()
