"""
Train an MLP on the full real training split and evaluate on test.
Uses resolve_preprocessed_dir (standard or fair / fair+OP).

Fair pipeline (--fair / --use_op) also reports ΔDP / ΔEO using binary S from the
base (non-OP) fair test CSV.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm, trange

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import cli
from dataset.loader_whole import DataLoaderCreator
from evaluation.eval_utils import compute_fairness, load_base_binary_sensitive
from model.model_utils import get_network
from utils import (
    get_time,
    is_fair_pipeline,
    measure_time,
    resolve_cctc_method_tag,
    set_seed,
    setup_logger,
)


class EvaluatorWhole:
    def __init__(self, args):
        self.args = args
        self.logger = args.logger
        self.method_tag = resolve_cctc_method_tag(self.args)
        self.compute_fair_metrics = is_fair_pipeline(self.args)

        self.save_path = os.path.join(
            _PROJECT_ROOT,
            "Results",
            "cctc_eval_whole",
            str(self.args.dataset),
            self.method_tag,
            f"{self.args.dataset}_{self.method_tag}_{self.args.eval_model}_"
            f"ep{self.args.epoch_eval_train}_bs{self.args.batch_train}_lr{self.args.lr_net}",
        )
        os.makedirs(self.save_path, exist_ok=True)

    def evaluate_whole(self):
        dl_creator = DataLoaderCreator(self.args)
        (
            trainloader,
            _valloader,
            testloader,
            num_classes,
            attr_name,
            numerical_feature_count,
            numerical_feature_idx,
            categorical_feature_count,
            categorical_feature_idx,
            unique_values_per_categorical_feature,
        ) = dl_creator.load_data()

        s_test = None
        if self.compute_fair_metrics:
            s_test = load_base_binary_sensitive(
                _PROJECT_ROOT, self.args.dataset, split="test"
            )
            print(
                f"[Eval_whole] Fair metrics ON (ΔDP/ΔEO); "
                f"S from base fair test |S=0|={(s_test == 0).sum()}, |S=1|={(s_test == 1).sum()}"
            )
        else:
            print("[Eval_whole] Fair metrics OFF (standard pipeline)")

        all_results = {
            "test_accuracy": [],
            "macro_f1": [],
            "delta_dp": [],
            "delta_eo": [],
        }

        for exp in trange(self.args.num_exp, desc="Experiments", unit="exp"):
            set_seed(exp)
            tqdm.write(f"\nStarting experiment {exp + 1}/{self.args.num_exp}...")

            net = get_network(
                self.args.eval_model,
                len(attr_name),
                num_classes,
                numerical_feature_count,
                numerical_feature_idx,
                categorical_feature_count,
                categorical_feature_idx,
                unique_values_per_categorical_feature,
                self.args.device,
            )
            criterion = nn.CrossEntropyLoss().to(self.args.device)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr_net)

            _, _ = self.train_model(trainloader, net, criterion, optimizer)

            test_loss, test_accuracy, macro_f1, y_pred, y_true = self.test_model(
                testloader, net, criterion
            )

            msg = (
                f"{get_time()} [Eval {exp}] "
                f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
                f"macro_f1: {macro_f1:.4f}"
            )
            if self.compute_fair_metrics:
                if len(s_test) != len(y_true):
                    raise ValueError(
                        f"S/test length mismatch: |S|={len(s_test)} |y|={len(y_true)}. "
                        "Base and eval test splits must be row-aligned."
                    )
                fair = compute_fairness(y_pred, y_true, s_test)
                delta_dp = round(float(fair["delta_dp"]), 6)
                delta_eo = round(float(fair["delta_eo"]), 6)
                all_results["delta_dp"].append(delta_dp)
                all_results["delta_eo"].append(delta_eo)
                msg += f", ΔDP: {delta_dp:.4f}, ΔEO: {delta_eo:.4f}"

            tqdm.write(msg)
            all_results["test_accuracy"].append(test_accuracy)
            all_results["macro_f1"].append(macro_f1)

        return all_results

    @measure_time
    def train_model(self, trainloader, net, criterion, optimizer):
        for epoch in trange(self.args.epoch_eval_train, desc="Epochs", unit="ep"):
            net.train()
            train_loss, correct, total = 0.0, 0, 0
            for features, labels in trainloader:
                features, labels = features.to(self.args.device), labels.to(self.args.device)
                outputs = net(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            epoch_loss = train_loss / total if total > 0 else 0.0
            epoch_acc = correct / total if total > 0 else 0.0

            if epoch % 5 == 0 or epoch == 0:
                tqdm.write(
                    f"Epoch {epoch}/{self.args.epoch_eval_train}: "
                    f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f} "
                )

    def test_model(self, test_loader, net, criterion):
        net.eval()
        test_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.args.device), labels.to(self.args.device)
                outputs = net(features)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        test_accuracy = correct / total if total > 0 else 0.0
        avg_loss = test_loss / total if total > 0 else 0.0
        all_labels = np.asarray(all_labels)
        all_preds = np.asarray(all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        return avg_loss, test_accuracy, macro_f1, all_preds, all_labels


def main():
    args = cli(standalone_mode=False)
    logger = setup_logger(name=f"{args.dataset}_eval_whole", log_file=None)
    args.logger = logger
    logger.info("==============Arguments===============")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    evaluator = EvaluatorWhole(args)
    all_results = evaluator.evaluate_whole()

    summary = {
        "test_accuracy": round(np.mean(all_results["test_accuracy"]), 4),
        "std_test_accuracy": round(np.std(all_results["test_accuracy"]), 4),
        "macro_f1": round(np.mean(all_results["macro_f1"]), 4),
        "std_macro_f1": round(np.std(all_results["macro_f1"]), 4),
    }
    if evaluator.compute_fair_metrics:
        summary["avg_delta_dp"] = round(np.mean(all_results["delta_dp"]), 4)
        summary["std_delta_dp"] = round(np.std(all_results["delta_dp"]), 4)
        summary["avg_delta_eo"] = round(np.mean(all_results["delta_eo"]), 4)
        summary["std_delta_eo"] = round(np.std(all_results["delta_eo"]), 4)

    final_output = {
        "Final Results (Average Over All Experiments)": summary,
    }

    json_path = os.path.join(evaluator.save_path, "final_results.json")
    with open(json_path, "w") as f:
        json.dump(final_output, f, indent=4)


if __name__ == "__main__":
    main()
