"""
Fair-CCTC: same condensation search as CCTC, with optional cluster reweighting (CR).
Output paths match CCTC.py (cctc_synthetic_output_dir / synthetic_csv_filename).
"""
import json
import os
import sys
import time
from collections import Counter

import faiss
import numpy as np
import pandas as pd
from tqdm import trange

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import cli
from dataset.loader_fair_cctc import FairCCTC_DataLoaderCreator
from utils import cctc_synthetic_output_dir, set_seed, synthetic_csv_filename


class CCTCCondense:
    def __init__(
        self,
        X_all,
        y_all,
        num_classes,
        reduction_rate,
        faiss_res,
        max_steps=100,
        gamma=1,
        device='CPU',
        use_gpu=True,
        sensitive_idx=0,
        sensitive_values=None,
        fair_cluster_reweight=False,
        fair_rho=1.0,
        fair_eps=1e-12,
    ):
        self.X_all = X_all
        self.y_all = y_all
        self.num_classes = num_classes
        self.reduction_rate = reduction_rate
        self.K = int(len(X_all) * reduction_rate)

        # precompute class-wise data
        self.class_data = {c: X_all[y_all == c] for c in range(num_classes)}

        self.k_max_c = {
            c: max(1, min(self.K - (self.num_classes - 1), self.class_data[c].shape[0]))
            for c in range(num_classes)
        }
        self.k_max = max(self.k_max_c.values())

        self.k_min = 1
        self.faiss_res = faiss_res
        # cache for k-means losses: shape [num_classes, k_max]
        self.loss_matrix = np.full((num_classes, self.k_max), np.inf, dtype=np.float64)
        self.max_steps = max_steps
        self.use_gpu = use_gpu
        self.gamma = gamma
        self.device = device
        self.sensitive_idx = sensitive_idx
        if sensitive_values is not None:
            self.sensitive_values = np.asarray(sensitive_values).reshape(-1).astype(int)
            if len(self.sensitive_values) != len(X_all):
                raise ValueError(
                    f"sensitive_values length {len(self.sensitive_values)} "
                    f"!= X_all rows {len(X_all)}"
                )
            self.class_sensitive_data = {
                c: self.sensitive_values[y_all == c] for c in range(num_classes)
            }
        else:
            self.sensitive_values = None
            self.class_sensitive_data = None
        self.fair_cluster_reweight = fair_cluster_reweight
        # Fairness intensity rho in [0, 1].
        # rho = 0: no reweighting (r=1);
        # rho = 1: minority and majority contribute equally to the centroid.
        self.fair_rho = float(fair_rho)
        if not (0.0 <= self.fair_rho <= 1.0):
            raise ValueError("fair_rho must be in [0, 1].")

        self.fair_eps = fair_eps
        self.epoch = 1
        self.src_list = []
        self.dst_list = []

    def _get_sensitive_values(self, X, row_sensitive=None):
        """Return binary sensitive values aligned with the rows of X."""
        if row_sensitive is not None:
            row_sensitive = np.asarray(row_sensitive).reshape(-1).astype(int)
            if len(row_sensitive) != len(X):
                raise ValueError(
                    f"row_sensitive length {len(row_sensitive)} != X rows {len(X)}"
                )
            return row_sensitive
        if self.sensitive_values is not None and len(self.sensitive_values) == len(X):
            return self.sensitive_values
        A = X[:, self.sensitive_idx]
        return np.rint(A).astype(int)

    def _cluster_sample_weights(self, X, assignments, row_sensitive=None):
        """
        Cluster-level debiasing via cluster reweighting (CR).

        For cluster S^i_j with minority / majority counts n_min, n_maj:
            r^i_j = 1 + rho * (n_maj / n_min - 1)
            w(x)  = r^i_j  if s = a_min
                  = 1      if s = a_maj

        Weights are later normalized implicitly when estimating the weighted
        centroid (divide by sum of weights in the cluster).

        rho = 0 -> no reweighting;
        rho = 1 -> minority and majority contribute equally.
        Pure clusters (only one sensitive group) keep weight 1.
        """
        A = self._get_sensitive_values(X, row_sensitive=row_sensitive)
        assignments = np.asarray(assignments).reshape(-1).astype(int)
        weights = np.ones(len(X), dtype=np.float64)

        if (not self.fair_cluster_reweight) or self.fair_rho == 0.0:
            return weights

        for cluster_id in np.unique(assignments):
            mask = assignments == cluster_id
            if not np.any(mask):
                continue

            A_c = A[mask]
            values, counts = np.unique(A_c, return_counts=True)
            # Need at least two sensitive groups to define min/maj.
            if len(values) < 2:
                continue

            min_idx = int(np.argmin(counts))
            maj_idx = int(np.argmax(counts))
            n_min = int(counts[min_idx])
            n_maj = int(counts[maj_idx])
            # Already balanced (including multi-way ties): leave weights at 1.
            if n_min == n_maj:
                continue

            a_min = int(values[min_idx])
            a_maj = int(values[maj_idx])
            r = 1.0 + self.fair_rho * (n_maj / float(n_min) - 1.0)

            # Minority upweighted; majority stays at 1; any middle groups stay at 1.
            weights[mask & (A == a_min)] = r
            weights[mask & (A == a_maj)] = 1.0

        return weights

    def _weighted_centroids(self, X, assignments, k, d, row_sensitive=None):
        """
        Recompute centroids using CR weights after FAISS KMeans.
        Normalization is done per cluster via weighted mean:
            centroid = sum_i w_i x_i / sum_i w_i
        """
        assignments = np.asarray(assignments).reshape(-1).astype(int)
        sample_weights = self._cluster_sample_weights(
            X, assignments, row_sensitive=row_sensitive,
        )
        centroids = np.zeros((k, d), dtype=np.float32)

        for cluster_id in range(k):
            mask = assignments == cluster_id
            if not np.any(mask):
                continue
            w = sample_weights[mask].reshape(-1, 1)
            denom = float(w.sum())
            if denom <= self.fair_eps:
                centroids[cluster_id] = X[mask].mean(axis=0)
            else:
                centroids[cluster_id] = (X[mask] * w).sum(axis=0) / denom

        return centroids, sample_weights

    def run_class_kmeans(self, X, k, niter=100, row_sensitive=None, kmeans_seed=42):
        """
        Run K-means clustering

        Args:
            X: input data with shape (n_samples, n_features)
            k: number of clusters
            niter: maximum number of iterations

        Returns:
            loss: clustering loss
            final_centroids: final cluster centroids
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if k <= 0:
            raise ValueError("k must be positive")

        k = min(k, len(X))

        X = np.ascontiguousarray(X, dtype='float32')
        n, d = X.shape

        clustering = faiss.Clustering(d, k)
        clustering.niter = niter
        clustering.seed = kmeans_seed
        clustering.min_points_per_centroid = 1
        clustering.nredo = 1
        clustering.verbose = False

        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = self.device.index if self.device.index is not None else 0
        gpu_index = faiss.GpuIndexFlatL2(self.faiss_res, d, cfg)

        try:
            clustering.train(X, gpu_index)

            final_centroids = faiss.vector_to_array(clustering.centroids).reshape(k, d).copy()

            D, I = gpu_index.search(X, 1)
            assignments = I[:, 0].astype(int)

            # Optional CR: r = 1 + rho*(n_maj/n_min - 1); minority gets r, majority gets 1.
            if self.fair_cluster_reweight:
                final_centroids, sample_weights = self._weighted_centroids(
                    X, assignments, k, d, row_sensitive=row_sensitive,
                )
                residual = X - final_centroids[assignments]
                ssd = (sample_weights * np.sum(residual ** 2, axis=1)).sum()
            else:
                ssd = D[:, 0].sum()

            weight = 1.0 / (n ** self.gamma)
            loss = ssd * weight
            return loss, final_centroids

        except Exception as e:
            print(f"K-means clustering failed: {e}")
            raise
        finally:
            del gpu_index

    def get_cost(self, solution):
        total = 0.0
        for c, k in enumerate(solution):
            idx = k - 1
            if self.loss_matrix[c, idx] == np.inf:
                Xc = self.class_data[c]
                sens_c = (
                    self.class_sensitive_data[c]
                    if self.class_sensitive_data is not None
                    else None
                )
                self.loss_matrix[c, idx], _ = self.run_class_kmeans(
                    Xc, k, row_sensitive=sens_c,
                )
            total += self.loss_matrix[c, idx]
        return total

    def src_2_des(self, current):
        for i in range(self.num_classes):
            if current[i] > self.k_min:
                self.src_list.append(i)
            if current[i] < self.k_max_c[i]:
                self.dst_list.append(i)

    def update_src_2_des(self, neighbor, c1, targets):
        """
        Update src_list and dst_list according to the current allocation in neighbor.
        c1: source class, where real_step is subtracted
        targets: target classes, either an int or a list[int]
        """
        if neighbor[c1] <= self.k_min:
            if c1 in self.src_list:
                self.src_list.remove(c1)
            if c1 not in self.dst_list:
                self.dst_list.append(c1)

        if isinstance(targets, int):
            targets = [targets]

        for c2 in targets:
            if neighbor[c2] >= self.k_max_c[c2]:
                if c2 in self.dst_list:
                    self.dst_list.remove(c2)
                if c2 not in self.src_list:
                    self.src_list.append(c2)

    def allocate_real_step(self, current, c1, real_step, k_max_c, dst_list):
        """
        First try room-proportional soft allocation; if nothing can be allocated,
        fall back to assigning all to one class.
        """
        neighbor = current.copy()
        neighbor[c1] -= real_step

        valid_receivers = [c for c in dst_list if c != c1 and current[c] < k_max_c[c]]
        if not valid_receivers:
            return None

        rooms = {c: k_max_c[c] - current[c] for c in valid_receivers}
        total_room = sum(rooms.values())
        if total_room == 0:
            return None

        alloc = {c: int(real_step * (room / total_room)) for c, room in rooms.items()}
        remaining = real_step - sum(alloc.values())

        if sum(alloc.values()) == 0:
            room_array = np.array([rooms[c] for c in valid_receivers])
            probs = room_array / room_array.sum()
            chosen_c = np.random.choice(valid_receivers, p=probs)
            neighbor[chosen_c] += real_step
            return neighbor

        sorted_cands = sorted(rooms.items(), key=lambda x: -x[1])
        for c, _ in sorted_cands:
            if remaining <= 0:
                break
            if current[c] + alloc[c] + 1 <= k_max_c[c]:
                alloc[c] += 1
                remaining -= 1

        for c, val in alloc.items():
            neighbor[c] += val

        return neighbor

    def cctc(self):
        init_dict = compute_num_class_dict(self.y_all, self.reduction_rate, False)
        current = [init_dict[c] for c in range(self.num_classes)]
        self.src_list.clear()
        self.dst_list.clear()
        self.src_2_des(current)
        best = current.copy()
        current_cost = self.get_cost(current)
        best_cost = current_cost
        std_dev = np.std(current)
        step = max(int(std_dev), 1)

        if not self.src_list or not self.dst_list:
            return init_dict, best, best_cost, self.epoch

        early_stop_threshold = 0.01
        early_stop_patience = 10
        small_delta_count = 0

        for it in range(self.max_steps):
            if not self.src_list or not self.dst_list:
                break

            c1 = np.random.choice(self.src_list)
            neighbor = current.copy()
            real_step = np.random.randint(1, step + 1)
            neighbor[c1] -= real_step
            if neighbor[c1] < self.k_min:
                continue
            neighbor = self.allocate_real_step(
                current=current,
                c1=c1,
                real_step=real_step,
                k_max_c=self.k_max_c,
                dst_list=self.dst_list,
            )
            if neighbor is None:
                continue

            cost_n = self.get_cost(neighbor)
            delta = cost_n - current_cost

            if delta < 0:
                targets = [c for c in range(self.num_classes) if neighbor[c] > current[c]]
                self.update_src_2_des(neighbor, c1, targets)
                current[:] = neighbor
                current_cost = cost_n
                step = max(step // 2, 1)

                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost
                accept = True
            else:
                accept = False

            if not accept and abs(delta) < early_stop_threshold:
                small_delta_count += 1
            else:
                small_delta_count = 0
            if small_delta_count >= early_stop_patience:
                self.epoch = it + 1
                return init_dict, best, best_cost, self.epoch
        self.epoch = it + 1
        return init_dict, best, best_cost, self.epoch


def compute_num_class_dict(labels, reduction_rate, balance):
    counter = Counter(labels)
    N = len(labels)
    C = len(counter)
    K = int(N * reduction_rate)

    num_class = {}
    if balance:
        per = max(K // C, 1)
        for c, cnt in counter.items():
            num_class[c] = min(per, cnt)
    else:
        for c, cnt in counter.items():
            num_class[c] = min(max(int(cnt * reduction_rate), 1), cnt)

    curr = sum(num_class.values())
    c_max = max(counter, key=lambda c: counter[c] - num_class[c])

    if curr < K:
        add = min(K - curr, counter[c_max] - num_class[c_max])
        num_class[c_max] += add
    elif curr > K:
        sub = min(curr - K, num_class[c_max] - 1)
        num_class[c_max] -= sub
    return num_class


def main():
    args = cli(standalone_mode=False)
    set_seed(42)
    args.method = "cctc"
    args.fair = True  # fair_CCTC always uses the fair preprocess / syn path tags
    dl_creator = FairCCTC_DataLoaderCreator(args)
    dst_train, num_classes, _ = dl_creator.load_train()
    s_train = dl_creator.load_base_binary_sensitive_train()

    X_all, y_all = dst_train.tensors
    X_all = X_all.numpy().astype("float32")
    y_all = y_all.numpy().astype("int32").flatten()
    args.num_classes = num_classes

    # Path tag in the same slot as CCTC's categorical_method: "fair" or "op".
    method_tag = dl_creator.fair_variant

    use_gpu = args.device.type == "cuda"
    gpu_res = faiss.StandardGpuResources() if use_gpu else None
    cctc_start = time.time()
    max_steps = 1000
    fair_cluster_reweight = True
    fair_rho = float(getattr(args, "fair_rho", 1.0))

    condense = CCTCCondense(
        X_all,
        y_all,
        num_classes,
        args.reduction_rate,
        gpu_res,
        max_steps=max_steps,
        gamma=args.gamma,
        device=args.device,
        use_gpu=use_gpu,
        sensitive_idx=0,
        sensitive_values=s_train,
        fair_cluster_reweight=fair_cluster_reweight,
        fair_rho=fair_rho,
    )
    init_dict, best_solution, best_cost, _real_epoch = condense.cctc()
    cctc_end = time.time()
    cctc_time = cctc_end - cctc_start

    results_dir = cctc_synthetic_output_dir(
        _PROJECT_ROOT,
        args.dataset,
        method_tag,
        args.reduction_rate,
        args.gamma,
    )
    os.makedirs(results_dir, exist_ok=True)

    select_times = []
    for seed in trange(args.num_exp, desc="Generate Runs"):
        set_seed(seed)
        start_time = time.time()
        all_centroids = []
        all_labels = []
        for c, k in enumerate(best_solution):
            sens_c = (
                condense.class_sensitive_data[c]
                if condense.class_sensitive_data is not None
                else None
            )
            _, centroids = condense.run_class_kmeans(
                condense.class_data[c],
                k,
                row_sensitive=sens_c,
                kmeans_seed=42 + seed,
            )
            all_centroids.append(centroids)
            all_labels.extend([c] * centroids.shape[0])
        merged = np.vstack(all_centroids)
        df = pd.DataFrame(merged, columns=[f"feat_{i}" for i in range(merged.shape[1])])
        df["target"] = all_labels
        elapsed = time.time() - start_time
        select_times.append(elapsed)
        csv_path = os.path.join(
            results_dir,
            synthetic_csv_filename(args.dataset, method_tag, args.reduction_rate, seed),
        )
        df.to_csv(csv_path, index=False)

    avg_time = np.mean(select_times)

    init_clean = {str(k): int(v) for k, v in init_dict.items()}
    output = {
        "initial_solution": init_clean,
        "best_solution": best_solution,
        "best_cost": best_cost,
        "cctc_time": cctc_time,
        "avg_generation_time_sec": avg_time,
    }
    json_path = os.path.join(results_dir, "results_summary.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
