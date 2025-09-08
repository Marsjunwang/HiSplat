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
    
class ECA(nn.Module):
    def __init__(self, k_size: int = 3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gap(x)              # [N,C,1,1]
        y = y.squeeze(-1).transpose(1, 2)  # [N,1,C]
        y = self.conv1d(y)
        y = self.sigmoid(y)
        y = y.transpose(1, 2).unsqueeze(-1)  # [N,C,1,1]
        return x * y

class ECAFusionReduce(nn.Module):
    def __init__(self, channels: int, k_size: int = 5):
        super().__init__()
        self.eca = ECA(k_size=k_size)
        self.reduce = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=1)
        x = self.eca(x)
        x = self.reduce(x)
        return self.bn(x)