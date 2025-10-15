import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
    
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

class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))
