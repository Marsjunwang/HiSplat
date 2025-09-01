import torch.nn as nn
import torch

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, 
                      dilation=2, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, 
                      dilation=4, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_fuse = nn.Conv2d(out_channels * 3, out_channels, 
                                   kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        fused = torch.cat((b1, b2, b3), dim=1)
        return self.conv_fuse(fused)

class ScaleHead(nn.Module):
    def __init__(self):
        super(ScaleHead, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=129*2, out_channels=128, 
                      kernel_size=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, 
                      padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.aspp = ASPP(in_channels=128, out_channels=128)
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 2) )

    def forward(self, x):
        x = self.features(x)
        x = self.aspp(x)  
        x = self.final(x)
        return x
