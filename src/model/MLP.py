import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, net_width, net_depth, net_act, net_norm, dropout_prob=0.0):
        super(MLP, self).__init__()
        self.layers = self._make_layers(input_dim, net_width, net_depth, net_norm, net_act, dropout_prob)
        self.classifier = nn.Linear(net_width, num_classes)

    def forward(self, x):
        out = self.layers(x)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.layers(x)
        return out

    def _get_activation(self, net_act):
        if net_act == "sigmoid":
            return nn.Sigmoid()
        if net_act == "relu":
            return nn.ReLU(inplace=True)
        if net_act == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.01)
        if net_act == "swish":
            return nn.SiLU()
        raise ValueError(f"unknown activation function: {net_act}")

    def _get_normlayer(self, net_norm, num_features):
        if net_norm == "batchnorm":
            return nn.BatchNorm1d(num_features)
        if net_norm == "layernorm":
            return nn.LayerNorm(num_features)
        if net_norm == "none":
            return None
        raise ValueError(f"unknown net_norm: {net_norm}")

    def _make_layers(self, input_dim, net_width, net_depth, net_norm, net_act, dropout_prob):
        layers = []
        in_features = input_dim
        for _ in range(net_depth):
            layers.append(nn.Linear(in_features, net_width))
            norm_layer = self._get_normlayer(net_norm, net_width)
            if norm_layer:
                layers.append(norm_layer)
            layers.append(self._get_activation(net_act))
            if dropout_prob > 0:
                layers.append(nn.Dropout(p=dropout_prob))
            in_features = net_width
        return nn.Sequential(*layers)


def get_default_mlp_setting():
    net_width = 64
    net_depth = 3
    net_act = "relu"
    net_norm = "batchnorm"
    dropout_prob = 0.1
    return net_width, net_depth, net_act, net_norm, dropout_prob
