import torch
import torch.nn as nn

def MLP(channels: list, do_bn=True, do_leaky=False, last_layer=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1 if last_layer else n):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            if do_leaky:
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class ConfidenceMLP(nn.Module):
    def __init__(self, feature_dim, in_dim, out_dim=1):
        super().__init__()
        self.feature_dim = feature_dim
        self.layers_f = MLP([feature_dim * 2, feature_dim * 2, feature_dim], last_layer=False)
        self.layers_c = MLP([in_dim, feature_dim, feature_dim], last_layer=False)
        self.layers = MLP([feature_dim, out_dim])
        
        nn.init.constant_(self.layers[-1].bias, 0.0)

    def forward(self, desc):
        inputs = torch.cat(desc[:-1], dim=-2)
        out_f = self.layers_f(inputs)
        out_c = self.layers_c(desc[-1])
        return torch.sigmoid(self.layers(out_f + out_c))