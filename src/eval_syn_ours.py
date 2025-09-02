import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.metrics import balanced_accuracy_score
from collections import defaultdict
from tqdm import tqdm, trange
from sklearn.metrics import f1_score  

from config import cli
from loader_syn_ours import SynDataLoaderCreator

class evaluator_syn:
    def __init__(self, args):
        self.args = args
        self.logger = args.logger  

        if self.args.method in ["ours"]:
            self.save_path_base = (
                f"Results/ours_results/{self.args.categorical_method}/{self.args.dataset}/{self.args.method}/{self.args.gamma}/"
                f"{self.args.reduction_rate}/"
                f"{self.args.dataset}_{self.args.method}_"
                f"{self.args.reduction_rate}_{self.args.eval_model}_{self.args.epoch_eval_train}_"
                f"{self.args.lr_net}"
            ) 

        os.makedirs(self.save_path_base, exist_ok=True)

    def evaluate_syn(self):
        aggregated_results = {}

        syn_dl_creator = SynDataLoaderCreator(self.args)
        (trainloader_list, valloader, testloader, num_classes, attr_name,
                numerical_feature_count, numerical_feature_idx,
                categorical_feature_count, categorical_feature_idx,
                unique_values_per_categorical_feature)= syn_dl_creator.load_syn_data()
        
        if not trainloader_list:
            self.logger.error("[Error] No synthetic datasets found. Exiting...")
            sys.exit(1)

        for exp in trange(len(trainloader_list),desc="Experiments", unit="exp"):
            exp_key = f"exp_{exp}"
            aggregated_results[exp_key] = []  

            for seed_i in range(self.args.num_exp):
                set_seed(seed_i)
                model = get_network(
                    self.args.eval_model, len(attr_name), num_classes,
                    numerical_feature_count, numerical_feature_idx, 
                    categorical_feature_count, categorical_feature_idx,
                    unique_values_per_categorical_feature, self.args.device
                )
                

                net = model
                net = net.to(self.args.device)

                lr = self.args.lr_net
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
                criterion = nn.CrossEntropyLoss().to(self.args.device)
                trainloader = trainloader_list[exp]

                train_time,(train_losses, train_accuracies) = self.train_model(trainloader, net, criterion, optimizer)

                test_loss, test_accuracy, class_accuracies, all_labels, all_preds, all_probs, macro_f1 = self.test_epoch(testloader, net, criterion)


                balanced_acc = balanced_accuracy_score(all_labels, all_preds)

                aggregated_results[exp_key].append({
                    "train_time": train_time,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "balanced_accuracy": balanced_acc,
                    "macro_f1": macro_f1,
                    "class_accuracies": class_accuracies 
                })
        
        combined = {"train_time": [], "test_loss": [], "test_accuracy": [], "balanced_accuracy": [],
                    "macro_f1": [], "class_accuracies": []}
        for exp_key, results_list in aggregated_results.items():
            for res in results_list:
                combined["train_time"].append(res["train_time"])
                combined["test_loss"].append(res["test_loss"])
                combined["test_accuracy"].append(res["test_accuracy"])
                combined["balanced_accuracy"].append(res["balanced_accuracy"])
                combined["macro_f1"].append(res["macro_f1"])
                combined["class_accuracies"].append(res["class_accuracies"])

        sum_dict = defaultdict(float)
        count_dict = defaultdict(int)
        for class_acc in combined["class_accuracies"]:
            for cls, acc in class_acc.items():
                sum_dict[cls] += acc
                count_dict[cls] += 1
        avg_class_acc = {cls: round(sum_dict[cls] / count_dict[cls], 4) for cls in sum_dict}

        overall_results = {
            "avg_train_time": round(np.mean(combined["train_time"]), 4),
            "avg_test_loss": round(np.mean(combined["test_loss"]), 4),
            "avg_test_accuracy": round(np.mean(combined["test_accuracy"]), 4),
            "std_test_accuracy": round(np.std(combined["test_accuracy"]), 4),
            "avg_balanced_accuracy": round(np.mean(combined["balanced_accuracy"]), 4),
            "std_balanced_accuracy": round(np.std(combined["balanced_accuracy"]), 4),
            "avg_macro_f1": round(np.mean(combined["macro_f1"]), 4),
            "std_macro_f1": round(np.std(combined["macro_f1"]), 4),
            "class_accuracies": avg_class_acc
        }

        return overall_results

    def train_model(self, trainloader, net, criterion, optimizer):
        train_losses = []
        train_accuracies = []

        for epoch in trange(self.args.epoch_eval_train,
                        desc="Epochs",
                        unit="ep"):
            net.train()
            train_loss, correct, total = 0.0, 0, 0
            for features, labels in trainloader:
                features, labels = features.to(self.args.device), labels.to(self.args.device)
                if self.args.eval_model == 'tabr':
                    outputs = net(features, labels)
                else:
                    outputs = net(features)                
                
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            train_losses.append(train_loss / total if total > 0 else 0.0)
            train_accuracies.append(correct / total if total > 0 else 0.0)

            if epoch % 5 == 0 or epoch == 0:
                self.logger.info(
                    f"Epoch {epoch}/{self.args.epoch_eval_train}: "
                    f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f} "
                )
                tqdm.write(
                    f"Epoch {epoch}/{self.args.epoch_eval_train}: "
                    f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f} "
                )
        return train_losses, train_accuracies
    
    def test_epoch(self, dataloader, net, criterion):
        net.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_preds = []
        all_probs = []

        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        with torch.no_grad():
            for features, labels in dataloader:
                features = features.float().to(self.args.device)
                labels = labels.long().to(self.args.device)
                batch_size = labels.size(0)

                if self.args.eval_model == 'tabr':
                    outputs = net(features, None)
                else:
                    outputs = net(features)
                loss = criterion(outputs, labels)

                predictions = torch.argmax(outputs, dim=-1)
                probabilities = torch.softmax(outputs, dim=1)

                total_loss += loss.item() * batch_size
                total_correct += (predictions == labels).sum().item()
                total_samples += batch_size

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())

                for label, pred in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        class_accuracies = {
            int(cls): class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
            for cls in class_total
        }
        return avg_loss, avg_acc, class_accuracies, all_labels, all_preds, all_probs, macro_f1

def main():
    args = cli(standalone_mode=False)
    evaluator = evaluator_syn(args)
    overall_eval_results = evaluator.evaluate_syn()
    final_output = {
    "Final Evaluation Results": overall_eval_results,
    }
    
    json_path = os.path.join(evaluator.save_path_base, "final_results.json")
    with open(json_path, "w") as f:
        json.dump(final_output, f, indent=4)
    
if __name__ == '__main__':
    main()
