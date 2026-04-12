import functools
import logging
import os
import random
import time

import numpy as np
import torch

loggers = {}


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def set_seed(seed):
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(name="experiment", log_file="experiment.log", overwrite=False):
    global loggers
    if name in loggers:
        return loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        mode = "w" if overwrite else "a"
        fh = logging.FileHandler(log_file, mode=mode)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
    loggers[name] = logger
    return logger


def measure_time(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        elapsed_time = time.time() - start_time
        return elapsed_time, result

    return wrapper


# --- Project layout (preprocessed data + CCTC outputs; keep writers/readers aligned) ---


def param_dirname(value):
    """Stable path segment for float CLI params (reduction_rate, gamma)."""
    f = float(value)
    if abs(f - round(f)) < 1e-9 and abs(f) < 1e15:
        return str(int(round(f)))
    text = f"{f:.8g}"
    return text if text not in ("-0", "-0.0") else "0"


def preprocessed_dataset_dir(project_root, categorical_method, dataset_name):
    """dataset/preprocessed_datasets/<categorical_method>/<dataset>/"""
    return os.path.join(
        project_root,
        "dataset",
        "preprocessed_datasets",
        categorical_method,
        dataset_name,
    )


def cctc_synthetic_output_dir(
    project_root,
    dataset_name,
    categorical_method,
    reduction_rate,
    gamma,
):
    """Directory where CCTC writes synthetic CSVs; SynDataLoaderCreator reads the same path."""
    return os.path.join(
        project_root,
        "Results",
        "cctc_datasets",
        dataset_name,
        categorical_method,
        param_dirname(reduction_rate),
        param_dirname(gamma),
    )


def synthetic_csv_filename(dataset_name, categorical_method, reduction_rate, seed):
    """One synthetic run; glob *seed{n}* still matches."""
    rr = param_dirname(reduction_rate)
    return f"{dataset_name}_{categorical_method}_r{rr}_seed{int(seed)}.csv"
