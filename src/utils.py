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


def fair_preprocessed_dataset_dir(project_root, dataset_name, use_op=False):
    """
    Fair preprocessed input leaf:
      dataset/preprocessed_datasets_fair/<dataset>/
      dataset/preprocessed_datasets_fair_op/<dataset>/
    """
    folder = "preprocessed_datasets_fair_op" if use_op else "preprocessed_datasets_fair"
    return os.path.join(project_root, "dataset", folder, dataset_name)


def is_fair_pipeline(args) -> bool:
    """True for fair / fair+OP runs (--fair or --use_op)."""
    return bool(getattr(args, "fair", False) or getattr(args, "use_op", False))


def resolve_cctc_method_tag(args) -> str:
    """
    Path segment under Results/cctc_datasets/<dataset>/<tag>/...
      standard: categorical_method (e.g. autoencoder)
      fair:     'fair'
      fair+OP:  'op'
    """
    if is_fair_pipeline(args):
        return "op" if getattr(args, "use_op", False) else "fair"
    return args.categorical_method


def resolve_preprocessed_dir(project_root, args) -> str:
    """
    Real preprocessed leaf used by loaders / eval:
      standard: dataset/preprocessed_datasets/<categorical_method>/<dataset>/
      fair:     dataset/preprocessed_datasets_fair/<dataset>/
      fair+OP:  dataset/preprocessed_datasets_fair_op/<dataset>/
    """
    if is_fair_pipeline(args):
        return fair_preprocessed_dataset_dir(
            project_root,
            args.dataset,
            use_op=bool(getattr(args, "use_op", False)),
        )
    return preprocessed_dataset_dir(
        project_root, args.categorical_method, args.dataset
    )


def resolve_cctc_synthetic_dir(project_root, args) -> str:
    """Results/cctc_datasets/<dataset>/<method_tag>/<rr>/<gamma>/"""
    return cctc_synthetic_output_dir(
        project_root,
        args.dataset,
        resolve_cctc_method_tag(args),
        args.reduction_rate,
        args.gamma,
    )


def cctc_synthetic_output_dir(
    project_root,
    dataset_name,
    categorical_method,
    reduction_rate,
    gamma,
):
    """Directory where CCTC / fair_CCTC write synthetic CSVs."""
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
