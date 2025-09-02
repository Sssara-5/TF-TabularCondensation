import os
import sys
import pandas as pd
import torch
from torch.utils.data import TensorDataset


from config import cli

class Ours_DataLoaderCreator:
    def __init__(self, args):
        self.args = args
        self.train_csv_path = (
            f"dataset/preprocessed_datasets/{self.args.categorical_method}/{self.args.dataset}/{self.args.dataset}_train.csv"
        )

    def load_ours(self):
        if os.path.exists(self.train_csv_path):
            print(f"[Train_data] Found Train Data CSV (Train):\n  {self.train_csv_path}")
            df_train = pd.read_csv(self.train_csv_path)
        else:
            print(f"[Train_data] Not Found Train Data CSV:\n  {self.train_csv_path}")
            sys.exit(1)

        num_classes = len(df_train['target'].unique())
        all_cols = df_train.columns 
        attr_name = list(all_cols[:-1])    
        train_attr = df_train.columns[:-1]
        X_train = df_train[train_attr].values
        y_train = df_train['target'].values

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        dst_train = TensorDataset(X_train_tensor, y_train_tensor)

        return dst_train, num_classes, attr_name


if __name__ == "__main__":
    args = cli(standalone_mode=False)
    dl_creator = Ours_DataLoaderCreator(args)
    dst_train, num_classes, attr_name = dl_creator.load_ours()

    print("[main] Train size:", len(dst_train))
    print("[main] num_classes:", num_classes)
    print("[main] attr_name:", attr_name)
