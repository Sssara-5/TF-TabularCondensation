import os
import time
import json
import numpy as np
import pandas as pd
import faiss
from tqdm import trange
from collections import Counter
import threading
import time
import numpy as np
import faiss
import pynvml
import random

from config import cli
from loader_ours import Ours_DataLoaderCreator

def init_nvml():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_used(handle):
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

class GpuMonitor:
    def __init__(self, interval=0.01, gpu_id=0):
        self.handle = init_nvml()
        self.interval = interval
        self.peak = 0
        self._running = False
        self._thread = None

    def _poll(self):
        while self._running:
            used = get_gpu_used(self.handle)
            if used > self.peak:
                self.peak = used
            time.sleep(self.interval)

    def start(self):
        self.peak = 0
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        return self.peak / 1024**2

class GreedyCondense:
    def __init__(self, X_all, y_all, num_classes, reduction_rate, faiss_res, max_steps=100,  gamma= 1, device='CPU', use_gpu=True):
        self.X_all = X_all
        self.y_all = y_all
        self.num_classes = num_classes
        self.reduction_rate = reduction_rate
        self.K = int(len(X_all) * reduction_rate)
        self.class_data = {c: X_all[y_all == c] for c in range(num_classes)}
        self.k_max_c = {
            c: max(1, min(self.K - (self.num_classes - 1), self.class_data[c].shape[0]))
            for c in range(num_classes)
        }
        self.k_max = max(self.k_max_c.values())
        self.k_min = 1
        self.faiss_res = faiss_res
        self.loss_matrix = np.full((num_classes, self.k_max), np.inf, dtype=np.float64)
        self.max_steps = max_steps
        self.use_gpu = use_gpu
        self.gamma = gamma
        self.device = device
        self.epoch = 1
        self.src_list=[]
        self.dst_list=[]


    def run_class_kmeans(self, X, k, niter=100):
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
        clustering.seed = 42
        clustering.min_points_per_centroid = 1
        clustering.nredo = 1
        clustering.verbose = False

        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = self.device.index if self.device.index is not None else 0
        gpu_index = faiss.GpuIndexFlatL2(self.faiss_res, d, cfg)

        try:
            clustering.train(X, gpu_index)
            final_centroids = faiss.vector_to_array(clustering.centroids).reshape(k, d).copy()
            D, _ = gpu_index.search(X, 1)
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
                self.loss_matrix[c, idx], _ = self.run_class_kmeans(Xc, k)            
            total += self.loss_matrix[c, idx] 
        return total

    def src_2_des(self,current):
        for i in range(self.num_classes):
            if current[i]>self.k_min:
                self.src_list.append(i)
            if current[i]<self.k_max_c[i]:
                self.dst_list.append(i)

    def update_src_2_des(self, neighbor, c1, targets):
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

    def greedy(self):
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
            real_step= np.random.randint(1, step + 1)
            neighbor[c1] -= real_step
            if neighbor[c1] < self.k_min:
                continue
            neighbor = self.allocate_real_step(
                current=current,
                c1=c1,
                real_step=real_step,
                k_max_c=self.k_max_c,
                dst_list=self.dst_list
            )
            if neighbor is None:
                continue 

            cost_n = self.get_cost(neighbor)
            delta = cost_n - current_cost

            if delta < 0:
                targets = [c for c in range(self.num_classes) if neighbor[c] > current[c]]
                self.update_src_2_des(neighbor, c1, targets)
                current      = neighbor
                current_cost = cost_n
                step = max(step // 2, 1)

                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost
                accept = True
            else:
                accept = False

            if not accept and cost_n < current_cost - early_stop_threshold:
                small_delta_count = 0
            else:
                small_delta_count += 1       
            if small_delta_count >= early_stop_patience:
                self.epoch = it +1
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
    args.method = "ours"
    dl_creator = Ours_DataLoaderCreator(args)
    dst_train, num_classes, _ = dl_creator.load_ours()

    X_all, y_all = dst_train.tensors
    X_all = X_all.numpy().astype('float32')
    y_all = y_all.numpy().astype('int32').flatten()
    args.num_classes = num_classes
    monitor = GpuMonitor(interval=0.01)
    monitor.start()
    use_gpu = (args.device.type == "cuda")
    gpu_res = faiss.StandardGpuResources() if use_gpu else None
    greedy_start = time.time()
    max_steps =1000
    condense = GreedyCondense(
        X_all, y_all, num_classes, args.reduction_rate, gpu_res, max_steps=max_steps, gamma=args.gamma, device=args.device,
        use_gpu=(args.device.type == "cuda")
    )
    init_dict, best_solution, best_cost,real_epoch = condense.greedy()
    greedy_end = time.time()
    greedy_time = greedy_end-greedy_start
    peak_gpu = monitor.stop()

    args.our_verison = "faiss"
    results_dir = os.path.join("Results", "ours_datasets", args.dataset, args.method, args.categorical_method, args.our_verison, f"{args.reduction_rate}",f"{args.gamma}")
    
    os.makedirs(results_dir, exist_ok=True)

    select_times = []
    
    for seed in trange(args.num_exp, desc='Generate Runs'):
        set_seed(seed)
        start_time = time.time()
        all_centroids = []
        all_labels = []
        for c, k in enumerate(best_solution):
            _, centroids = condense.run_class_kmeans(condense.class_data[c], k)
            all_centroids.append(centroids)
            all_labels.extend([c] * centroids.shape[0])
        merged = np.vstack(all_centroids)
        df = pd.DataFrame(merged, columns=[f'feat_{i}' for i in range(merged.shape[1])])
        df['target'] = all_labels
        elapsed = time.time() - start_time
        select_times.append(elapsed)
        csv_path = os.path.join(results_dir, f"{args.dataset}_{args.categorical_method}_{args.reduction_rate}_seed{seed}.csv")
        df.to_csv(csv_path, index=False)

    avg_time = np.mean(select_times)
    init_clean = { str(k): int(v) for k, v in init_dict.items() }
    output = {
        "initial solution":init_clean,
        "best_solution": best_solution,
        "best_cost": best_cost,
        "greedy_time": greedy_time,
        "avg_generation_time_sec": avg_time,
        "peak_gpu_memory_mb": peak_gpu,
        "real_epoch":real_epoch
    }
    json_path = os.path.join(results_dir, "results_summary.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()
