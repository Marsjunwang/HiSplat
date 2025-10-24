# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from ..utils.model import ECAFusionReduce
from .attn.C2PSA import C2PSA

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, 
                 block, 
                 layers, 
                 num_classes=1000, 
                 num_input_images=1, 
                 channels=3,
                 eca_fusion_reduce=False,
                 homo_encoder=False):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if not eca_fusion_reduce and not homo_encoder:
            self.inplanes = 128 * num_input_images
        elif eca_fusion_reduce and not homo_encoder:
            self.eca_fusion_reduce = ECAFusionReduce(channels=128)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def resnet_multiimage_input(num_layers, 
                            pretrained=False, 
                            num_input_images=1, 
                            channels=3,
                            eca_fusion_reduce=False,
                            homo_encoder=False):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 34, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 34: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(
        block_type, 
        blocks, 
        num_input_images=num_input_images, 
        channels=channels,
        eca_fusion_reduce=eca_fusion_reduce,
        homo_encoder=homo_encoder)

    if pretrained:
        from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
        weights_dict = {18: ResNet18_Weights.IMAGENET1K_V1, 34: ResNet34_Weights.IMAGENET1K_V1, 50: ResNet50_Weights.IMAGENET1K_V1}
        weights = weights_dict[num_layers]
        loaded = torch.hub.load_state_dict_from_url(weights.url)
        if not eca_fusion_reduce and not homo_encoder:
            loaded["layer3.0.conv1.weight"] = torch.cat([
                    loaded["layer3.0.conv1.weight"]] * num_input_images, 1
                    ) / num_input_images
            loaded["layer3.0.downsample.0.weight"] = torch.cat([
                    loaded["layer3.0.downsample.0.weight"]] * num_input_images, 1
                    ) / num_input_images
        if channels == 3:
            pass
        else:
            print(f"RESNET_HIE: input channels: {channels}, output channels: 64")
            conv1_weight = torch.empty(64, channels, 7, 7)
            nn.init.kaiming_normal_(conv1_weight, mode='fan_out', nonlinearity='relu')
            loaded['conv1.weight'] = conv1_weight
            
        model.load_state_dict(loaded, strict=False)
        
    return model

class Normalize(nn.Module):
    def __init__(self, mean=0.45, std=0.225):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x):
        return (x - self.mean) / self.std
    
class ResnetHierarchicalEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, 
                 num_layers, 
                 pretrained, 
                 num_input_images=1, 
                 channels=3, 
                 eca_fusion_reduce=False,
                 homo_encoder=False,
                 spatila_softmax=True,
                 spatila_softmax_tau=0.2,
                 use_norm_xy=False,
                 attn_encoder=None,
                 **kwargs):
        super(ResnetHierarchicalEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.channels = channels
        if channels == 3:
            self.input_normlize = Normalize()
        else:
            self.input_normlize = nn.Identity()
            
        self._eca_fusion_reduce = eca_fusion_reduce
        if eca_fusion_reduce:
            self.eca_fusion_reduce = ECAFusionReduce(channels=128)
        self._homo_encoder = homo_encoder
        
        self._spatila_softmax = spatila_softmax
        # Register tau as a proper leaf Parameter to avoid holding a graph across steps
        if spatila_softmax: 
            self._spatila_softmax_tau = nn.Parameter(
                torch.tensor(spatila_softmax_tau, device="cuda").float(), 
                requires_grad=True)
        
        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(
                num_layers, 
                pretrained, 
                num_input_images, 
                channels=channels,
                eca_fusion_reduce=eca_fusion_reduce,
                homo_encoder=homo_encoder)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
            
        self.use_norm_xy = use_norm_xy
        if use_norm_xy:
            # 1. unfold norm_xy encoder
            self.norm_xy_encoder = nn.Sequential(
                nn.Conv2d(128+128, 128, kernel_size=1, padding=0, bias=False),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False, groups=128),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=1, padding=0, bias=False),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False, groups=128),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            # # 2. interpolate norm_xy encoder
            # self.norm_xy_encoder = nn.Sequential(
            #     nn.Conv2d(128+2, 128, kernel_size=1, padding=0, bias=False),
            #     nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False, groups=128),
            #     nn.BatchNorm2d(128),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(128, 128, kernel_size=1, padding=0, bias=False),
            #     nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False, groups=128),
            #     nn.BatchNorm2d(128),
            #     nn.ReLU(inplace=True),
            # )

        self.attn_encoder = None
        if attn_encoder:
            self.attn_encoder = C2PSA(*attn_encoder)

    def forward(self, input_image, norm_xy=None):
        self.features = []
        x = self.input_normlize(input_image)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        feat_layer2 = self.encoder.layer2(self.features[-1])
        self.features.append(feat_layer2)
        B2, C, H, W = feat_layer2.shape
        B = B2 // 2
        if self._spatila_softmax:
            spatila_softmax_feat = F.softmax(
                (feat_layer2 / self._spatila_softmax_tau).view(B, 2, C, -1), 
                dim=-1).view(B2, C, H, W)
            feat_layer2 = feat_layer2 * spatila_softmax_feat + feat_layer2
            
        if self.use_norm_xy:
            with torch.no_grad():
                # 1. unfold norm_xy
                scale = input_image.shape[-1] // feat_layer2.shape[-1]
                norm_xy = norm_xy.unfold(2, scale, scale
                                         ).unfold(3, scale, scale)
                norm_xy = norm_xy.permute(0, 4, 5, 1, 2, 3
                                          ).reshape(B2, -1, H, W)
                # # 2. interpolate norm_xy to feat_layer2.shape
                # norm_xy = F.interpolate(norm_xy, size=(H, W), mode='bilinear')

            feat_layer2 = self.norm_xy_encoder(
                torch.cat([feat_layer2, norm_xy], dim=1)) + feat_layer2
            
        if not self._eca_fusion_reduce and not self._homo_encoder:
            self.features.append(
                self.encoder.layer3(feat_layer2.reshape(B, 2*C, H, W)))
        elif self._eca_fusion_reduce and not self._homo_encoder:
            feat_layer2 = feat_layer2.view(B, 2, C, H, W)
            self.features.append(
                self.encoder.layer3(
                    self.eca_fusion_reduce(feat_layer2[:,0], 
                                           feat_layer2[:,1])))
        self.features.append(self.encoder.layer4(self.features[-1]))

        if self.attn_encoder:
            self.features.append(self.attn_encoder(self.features[-1]))

        return self.features
