"""Build the evaluation MLP."""
import torch

from model.MLP import MLP, get_default_mlp_setting


def get_network(
    model_name,
    num_features,
    num_classes,
    _numerical_feature_count,
    _numerical_feature_idx,
    _categorical_feature_count,
    _categorical_feature_idx,
    _unique_values_per_categorical_feature,
    device,
):
    if model_name == "MLP":
        net_width, net_depth, net_act, net_norm, dropout_prob = get_default_mlp_setting()
        net = MLP(
            input_dim=num_features,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            dropout_prob=dropout_prob,
        )
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    return net.to(device if isinstance(device, torch.device) else torch.device(device))
