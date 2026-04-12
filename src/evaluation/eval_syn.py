"""
Train an MLP on CCTC synthetic CSVs and evaluate on the real preprocessed test set.
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
from model.model_utils import get_network
from utils import get_time, measure_time, param_dirname, set_seed, setup_logger


class EvaluatorSyn:
    def __init__(self, args):
        self.args = args
        self.logger = args.logger
        if self.args.method != "cctc":
            self.logger.error("eval_syn expects --method cctc (paths match CCTC outputs).")
            sys.exit(1)

        self.save_path_base = os.path.join(
            _PROJECT_ROOT,
            "Results",
            "cctc_eval_syn",
            self.args.categorical_method,
            self.args.dataset,
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

        combined = {
            "test_accuracy": [],
            "macro_f1": [],
        }
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
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
                criterion = nn.CrossEntropyLoss().to(self.args.device)
                trainloader = trainloader_list[exp]

                _, _ = self.train_model(trainloader, net, criterion, optimizer)

                test_accuracy, macro_f1 = self.test_epoch(testloader, net)

                tqdm.write(
                    f"{get_time()} [Exp {exp} Seed {seed_i}] Training finished. "
                    f"Test Accuracy: {test_accuracy:.4f}, Macro-F1: {macro_f1:.4f}"
                )

                combined["test_accuracy"].append(test_accuracy)
                combined["macro_f1"].append(macro_f1)

        overall_results = {
            "avg_test_accuracy": round(np.mean(combined["test_accuracy"]), 4),
            "std_test_accuracy": round(np.std(combined["test_accuracy"]), 4),
            "avg_macro_f1": round(np.mean(combined["macro_f1"]), 4),
            "std_macro_f1": round(np.std(combined["macro_f1"]), 4),
        }
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
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        return avg_acc, macro_f1


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
