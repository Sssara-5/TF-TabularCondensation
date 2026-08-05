"""
Train an MLP on CCTC synthetic CSVs and evaluate on the real preprocessed test set.

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
from dataset.loader_syn import SynDataLoaderCreator
from evaluation.eval_utils import compute_fairness, load_base_binary_sensitive
from model.model_utils import get_network
from utils import (
    get_time,
    is_fair_pipeline,
    measure_time,
    param_dirname,
    resolve_cctc_method_tag,
    set_seed,
    setup_logger,
)


class EvaluatorSyn:
    def __init__(self, args):
        self.args = args
        self.logger = args.logger
        if self.args.method != "cctc":
            self.logger.error("eval_syn expects --method cctc (paths match CCTC outputs).")
            sys.exit(1)

        self.method_tag = resolve_cctc_method_tag(self.args)
        self.compute_fair_metrics = is_fair_pipeline(self.args)
        # Same leaf order as eval_whole: .../<dataset>/<method_tag>/...
        self.save_path_base = os.path.join(
            _PROJECT_ROOT,
            "Results",
            "cctc_eval_syn",
            self.args.dataset,
            self.method_tag,
            param_dirname(self.args.reduction_rate),
            param_dirname(self.args.gamma),
            f"{self.args.dataset}_{self.args.method}_r{param_dirname(self.args.reduction_rate)}_"
            f"{self.args.eval_model}_ep{self.args.epoch_eval_train}_lr{self.args.lr_net}",
        )
        os.makedirs(self.save_path_base, exist_ok=True)

    def evaluate_syn(self):
        syn_dl_creator = SynDataLoaderCreator(self.args)
        (
            trainloader_list,
            _valloader,
            testloader,
            num_classes,
            attr_name,
            numerical_feature_count,
            numerical_feature_idx,
            categorical_feature_count,
            categorical_feature_idx,
            unique_values_per_categorical_feature,
        ) = syn_dl_creator.load_syn_data()
        print(f"[Eval_syn] Number of synthetic datasets (seeds) loaded: {len(trainloader_list)}")
        if not trainloader_list:
            self.logger.error("[Error] No synthetic datasets found. Exiting...")
            sys.exit(1)

        s_test = None
        if self.compute_fair_metrics:
            s_test = load_base_binary_sensitive(
                _PROJECT_ROOT, self.args.dataset, split="test"
            )
            print(
                f"[Eval_syn] Fair metrics ON (ΔDP/ΔEO); "
                f"S from base fair test |S=0|={(s_test == 0).sum()}, |S=1|={(s_test == 1).sum()}"
            )
        else:
            print("[Eval_syn] Fair metrics OFF (standard pipeline)")

        combined = {
            "test_accuracy": [],
            "macro_f1": [],
            "delta_dp": [],
            "delta_eo": [],
        }
        # Outer: each synthetic CSV; inner: different MLP init seeds per CSV.
        for exp in trange(len(trainloader_list), desc="Experiments", unit="exp"):
            for seed_i in range(self.args.num_exp):
                set_seed(seed_i)
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
                lr = self.args.lr_net
                optimizer = torch.optim.SGD(
                    net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
                )
                criterion = nn.CrossEntropyLoss().to(self.args.device)
                trainloader = trainloader_list[exp]

                _, _ = self.train_model(trainloader, net, criterion, optimizer)

                test_accuracy, macro_f1, y_pred, y_true = self.test_epoch(testloader, net)

                msg = (
                    f"{get_time()} [Exp {exp} Seed {seed_i}] "
                    f"Test Accuracy: {test_accuracy:.4f}, Macro-F1: {macro_f1:.4f}"
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
                    combined["delta_dp"].append(delta_dp)
                    combined["delta_eo"].append(delta_eo)
                    msg += f", ΔDP: {delta_dp:.4f}, ΔEO: {delta_eo:.4f}"

                tqdm.write(msg)
                combined["test_accuracy"].append(test_accuracy)
                combined["macro_f1"].append(macro_f1)

        overall_results = {
            "avg_test_accuracy": round(np.mean(combined["test_accuracy"]), 4),
            "std_test_accuracy": round(np.std(combined["test_accuracy"]), 4),
            "avg_macro_f1": round(np.mean(combined["macro_f1"]), 4),
            "std_macro_f1": round(np.std(combined["macro_f1"]), 4),
        }
        if self.compute_fair_metrics:
            overall_results["avg_delta_dp"] = round(np.mean(combined["delta_dp"]), 4)
            overall_results["std_delta_dp"] = round(np.std(combined["delta_dp"]), 4)
            overall_results["avg_delta_eo"] = round(np.mean(combined["delta_eo"]), 4)
            overall_results["std_delta_eo"] = round(np.std(combined["delta_eo"]), 4)
        return overall_results

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
                self.logger.info(
                    f"Epoch {epoch}/{self.args.epoch_eval_train}: "
                    f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f} "
                )
                tqdm.write(
                    f"Epoch {epoch}/{self.args.epoch_eval_train}: "
                    f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f} "
                )

    def test_epoch(self, dataloader, net):
        net.eval()
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for features, labels in dataloader:
                features = features.float().to(self.args.device)
                labels = labels.long().to(self.args.device)
                outputs = net(features)
                predictions = torch.argmax(outputs, dim=-1)

                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())

        avg_acc = total_correct / total_samples
        all_labels = np.asarray(all_labels)
        all_preds = np.asarray(all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        return avg_acc, macro_f1, all_preds, all_labels


def main():
    args = cli(standalone_mode=False)
    logger = setup_logger(name=f"{args.dataset}_eval_syn", log_file=None)
    args.logger = logger
    logger.info("==============Arguments===============")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    evaluator = EvaluatorSyn(args)
    overall_eval_results = evaluator.evaluate_syn()

    final_output = {
        "Final Evaluation Results": overall_eval_results,
    }

    json_path = os.path.join(evaluator.save_path_base, "final_results.json")
    with open(json_path, "w") as f:
        json.dump(final_output, f, indent=4)


if __name__ == "__main__":
    main()
