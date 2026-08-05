"""Shared CLI for CCTC, preprocessing helpers, and evaluation scripts."""
import json
import os
import sys
from pprint import pformat
from typing import Optional

import click
import torch

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Default categorical encoding per dataset (used when --categorical_method is omitted).
# Datasets not listed here are assumed to have no categorical columns: default to "na"
# (numerical scaling/split only; see pre_processing.process_categorical).
DATASET_CATEGORICAL_METHOD_DEFAULT = {
    "Adult": "autoencoder",
    "Diabetes130US": "autoencoder",
    "airlines": "autoencoder",
    "road-safety": "label_encoder",
    "Covertype": "label_encoder",
    "electricity": "target_encoder",
}

_VALID_CATEGORICAL_METHODS = frozenset(
    {"autoencoder", "label_encoder", "target_encoder", "na"}
)


def resolve_categorical_method(dataset: str, explicit: Optional[str]) -> str:
    if explicit is not None:
        if explicit not in _VALID_CATEGORICAL_METHODS:
            raise ValueError(
                f"categorical_method must be one of {sorted(_VALID_CATEGORICAL_METHODS)}, got {explicit!r}"
            )
        return explicit
    return DATASET_CATEGORICAL_METHOD_DEFAULT.get(dataset, "na")


class Obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

    def __repr__(self):
        return pformat(self.__dict__, compact=True)


def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=Obj)


@click.command()
@click.option("--dataset", default="Adult", show_default=True, help="Dataset name.")
@click.option(
    "--method",
    default="cctc",
    show_default=True,
    help="Method tag (use 'cctc' for condensation outputs and eval_syn paths).",
)
@click.option(
    "--reduction_rate",
    default=0.001,
    type=float,
    show_default=True,
    help="Target fraction of training samples kept as condensed prototypes (e.g. 0.001).",
)
@click.option(
    "--gamma",
    default=0.25,
    type=float,
    show_default=True,
    help="Penalty weight (gamma) in CCTC.",
)
@click.option(
    "--categorical_method",
    default=None,
    type=str,
    help=(
        "Categorical encoding: autoencoder, label_encoder, target_encoder, na. "
        "If omitted, uses dataset-specific default; unlisted datasets default to na (numeric-only)."
    ),
)
@click.option("--num_exp", default=5, type=int, show_default=True, help="Number of synthetic CSV runs.")
@click.option(
    "--fair",
    is_flag=True,
    default=False,
    help="Use fair pipeline paths (preprocessed_datasets_fair[_op], method tag fair/op).",
)
@click.option(
    "--use_op",
    is_flag=True,
    default=True,
    help="Fair-CCTC: use OP features (preprocessed_datasets_fair_op, method tag 'op'). "
    "Fair-CCTC includes OP by default in run_fair_pipeline.sh.",
)
@click.option(
    "--fair_rho",
    default=1.0,
    type=float,
    show_default=True,
    help="fair_CCTC: cluster-reweight intensity in [0, 1].",
)

@click.option(
    "--eval_model",
    default="MLP",
    show_default=True,
    help="Classifier for evaluation (MLP).",
)
@click.option(
    "--epoch_eval_train",
    default=100,
    type=int,
    show_default=True,
    help="Training epochs when evaluating on syn or whole data.",
)
@click.option(
    "--lr_net",
    default=0.001,
    type=float,
    show_default=True,
    help="Learning rate for the evaluation classifier.",
)
@click.option(
    "--batch_train",
    default=512,
    type=int,
    show_default=True,
    help="Batch size for val/test loaders (and whole-dataset train).",
)

@click.option("--device", default="0", show_default=True, help="CUDA device index.")
@click.pass_context
def cli(ctx, **kwargs):
    kwargs["categorical_method"] = resolve_categorical_method(
        kwargs["dataset"], kwargs.get("categorical_method")
    )
    # --use_op is fair-only; treat it as enabling the fair pipeline.
    if kwargs.get("use_op"):
        kwargs["fair"] = True
    args = dict2obj(kwargs)
    device_id = int(args.device)
    assert device_id < torch.cuda.device_count(), f"Invalid device ID {device_id}"
    print(f"Using GPU: {device_id} - {torch.cuda.get_device_name(device_id)}")
    args.device = torch.device(f"cuda:{device_id}")
    print("==============Arguments===============")
    print(args)
    return args


if __name__ == "__main__":
    cli()
