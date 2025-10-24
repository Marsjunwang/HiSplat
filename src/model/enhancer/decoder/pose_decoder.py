# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, 
                 num_frames_to_predict_for=None, 
                 stride=1, 
                 joint_pose=False,
                 num_view_per_frame=1,
                 **kwargs):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.joint_pose = joint_pose
        self.num_view_per_frame = num_view_per_frame

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU(inplace=False)

        self.net = nn.ModuleList(list(self.convs.values()))
        self.net.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        B, C, H, W = input_features[-1].shape
        if self.joint_pose:
            last_features = input_features[-1].reshape(-1, self.num_view_per_frame, C, H, W).mean(1)
        else:
            last_features = input_features[-1]

        cat_features = self.relu(self.convs["squeeze"](last_features))

        out = cat_features

        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = out.view(-1, self.num_frames_to_predict_for, 1, 6)
        axisangle = 0.1 * out[..., :3]
        translation = 0.1 * out[..., 3:]

        return axisangle, translation

class PoseDecoderV2(nn.Module):
    """
    Pose decoder with translation and rotation heads
    """
    def __init__(self, num_ch_enc, num_input_features, 
                 num_frames_to_predict_for=None, 
                 stride=1, 
                 joint_pose=False,
                 num_view_per_frame=1,
                 **kwargs):
        super(PoseDecoderV2, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.joint_pose = joint_pose
        self.num_view_per_frame = num_view_per_frame

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        # Trunk keeps a fixed 256-channel pipeline; heads branch from trunk
        self.convs[("pose", 0)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        # Rotation and translation heads (1x1) over the shared trunk
        self.convs[("pose", 2)] = nn.Conv2d(256, 3 * num_frames_to_predict_for, 1)
        self.convs[("pose", 3)] = nn.Conv2d(256, 3 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU(inplace=False)

        self.net = nn.ModuleList(list(self.convs.values()))
        self.net.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        B, C, H, W = input_features[-1].shape
        if self.joint_pose:
            last_features = input_features[-1].reshape(-1, self.num_view_per_frame, C, H, W).mean(1)
        else:
            last_features = input_features[-1]

        cat_features = self.relu(self.convs["squeeze"](last_features))

        # Shared trunk
        trunk = self.relu(self.convs[("pose", 0)](cat_features))
        trunk = self.relu(self.convs[("pose", 1)](trunk))

        # Heads
        rot_map = self.convs[("pose", 2)](trunk)
        trans_map = self.convs[("pose", 3)](trunk)

        # Global average pool to vectors
        rot_vec = rot_map.mean(3).mean(2)
        trans_vec = trans_map.mean(3).mean(2)

        axisangle = 0.1 * rot_vec.view(-1, self.num_frames_to_predict_for, 1, 3)
        translation = 0.1 * trans_vec.view(-1, self.num_frames_to_predict_for, 1, 3)

        return axisangle, translation
     
class PoseDecoderHomo(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, 
                 num_frames_to_predict_for=None, 
                 stride=1, 
                 joint_pose=False,
                 num_view_per_frame=1,
                 **kwargs):
        super(PoseDecoderHomo, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.joint_pose = joint_pose
        self.num_view_per_frame = num_view_per_frame

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(2, 128, 1)
        # two conv blocks with residual and dropout after activation
        self.convs[("pose", 0)] = nn.Conv2d(128, 128, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(128, 128, 3, stride, 1)

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout2d(p=0.2)

        self.net = nn.ModuleList(list(self.convs.values()))
        self.net.apply(self._init_weights)
        # replace conv heads with linear heads on pooled features for numerical stability
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.trans_head = nn.Linear(128, 3)
        self.rot_head = nn.Linear(128, 3)
        self.scale_head = nn.Linear(128, 1)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        B, C, H, W = input_features[-1].shape
        if self.joint_pose:
            last_features = input_features[-1].reshape(-1, self.num_view_per_frame, C, H, W).mean(1)
        else:
            last_features = input_features[-1]

        cat_features = self.relu(self.convs["squeeze"](last_features))

        out = cat_features

        # residual block stack
        for i in range(2):
            residual = out
            out = self.convs[("pose", i)](out)
            out = self.relu(out)
            out = self.dropout(out)
            # match shapes (they are equal here)
            out = out + residual

        # global average pool to vector
        out = self.global_pool(out).view(out.size(0), -1)
        
        axisangle = 0.1 * self.rot_head(out).view(-1, 
            1, 1, 3)
        translation = 0.1 * self.trans_head(out).view(-1, 
            1, 1, 3)
        scale = 0.1 * self.scale_head(out).view(-1, 
            1, 1, 1)

        return axisangle, translation * scale
