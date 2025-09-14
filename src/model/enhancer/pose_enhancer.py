from numpy import AxisError
from .enhancer import PoseEnhancer
from dataclasses import dataclass
from typing import List, Dict, Sequence
import os

from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F
import torch
import contextlib

from .encoder import ResnetEncoder, ResnetHierarchicalEncoder
from .decoder import PoseDecoder, PoseCNN
from .decoder.pose_decoder_sparse import PoseDecoderSparse
from .encoder.xfeat_sparse_encoder import XFeatSparseEncoder
from einops import rearrange
from .utils.transformation_pose import transformation_from_parameters
from .utils.pose_alignment import (align_world_to_view0, relative_pose_0_to_1, 
                                   split_pred_relative_two_directions)


@dataclass
class PoseEnhancerCfg:
    name: str
    input_data: str
    use_norm_xy: bool
    limit_pose_to_fov_overlap: bool
    fov_overlap_epsilon_deg: float
    pose_encoder: dict
    pose_decoder: dict

class PoseSeparateEnhancer(PoseEnhancer):
    def __init__(self, cfg: PoseEnhancerCfg):
        super().__init__(cfg)
        norm_xy_channels = 2 if cfg.use_norm_xy else 0
        cfg.pose_encoder.update(channels=cfg.pose_encoder["channels"] + norm_xy_channels)
        self._init_model(cfg)

        if self.cfg.input_data == "image":
            assert self.pose_encoder.channels == 3 + norm_xy_channels
        elif self.cfg.input_data == "feat":
            assert self.pose_encoder.channels == 128 + norm_xy_channels
        elif self.cfg.input_data == "feat_backbone":
            assert self.pose_encoder.channels == 32 + norm_xy_channels

        self.limit_pose_to_fov_overlap = cfg.limit_pose_to_fov_overlap
        self.fov_overlap_epsilon_deg = cfg.fov_overlap_epsilon_deg
    
    def _init_model(self, cfg: PoseEnhancerCfg):
        self.pose_encoder = ResnetEncoder(**cfg.pose_encoder)
        cfg.pose_decoder.update(num_ch_enc=self.pose_encoder.num_ch_enc)
        self.pose_decoder = PoseDecoder(**cfg.pose_decoder)
    
    def _get_RT_matrix(self, 
                       axisangle, 
                       translation, 
                       invert=False, 
                       intrinsics=None,
                       near=None,
                       far=None,
                       image_size=None):
        if False:
            from src.model.enhancer.utils.pose_overlap_viz import visualize_fov_overlap, export_fov_overlap_obj
            img = visualize_fov_overlap(axisangle, 
                                        translation, 
                                        intrinsics, 
                                        invert=False, 
                                        epsilon_deg=0.5, 
                                        resolution=256,
                                        image_size=image_size,
                                        near=near,
                                        far=far,
                                        save_to="fov_overlap.png",
                                        pair_mode="both")
            export_fov_overlap_obj(axisangle, translation, intrinsics, "fov_overlap.obj", invert=False, epsilon_deg=0.5)
            exit()
        transformation = transformation_from_parameters(axisangle, translation, 
                                                        invert=invert, intrinsics=intrinsics,
                                                        near=near,
                                                        far=far,
                                                        image_size=image_size,
                                                        limit_pose_to_fov_overlap=self.limit_pose_to_fov_overlap,
                                                        fov_overlap_epsilon_deg=self.fov_overlap_epsilon_deg)
        return transformation
    
    def forward(
        self,
        context: dict,
        features: Sequence) -> tuple[Dict, Sequence]:
        if self.cfg.input_data == "image":
            input_data = rearrange(context["image"], "b v c h w -> b (v c) h w")
        elif self.cfg.input_data == "feat":  # feat from transformer b v 128 64 64
            input_data = rearrange(features[1], "b v c h w -> b (v c) h w")
        elif self.cfg.input_data == "feat_backbone":  # feat from backbone (b v) 128 32 32
            input_data = rearrange(features[0][-1], "(b v) c h w -> b (v c) h w", b=context["image"].shape[0])
        else:
            raise ValueError(f"Invalid input_data: {self.cfg.input_data}")

        if self.cfg.use_norm_xy:
            norm_xy = rearrange(context["norm_xy"], "b v c h w -> b (v c) h w")
            if norm_xy.shape[-2:] != input_data.shape[-2:]:
                norm_xy = F.interpolate(norm_xy, size=input_data.shape[-2:], mode="bilinear", align_corners=False)
            input_data = torch.cat([input_data, norm_xy], dim=1)
        pose_feature = self.pose_encoder(input_data)
        axisangle, translation = self.pose_decoder(pose_feature)

        # Build SE3 with optional FOV overlap constraint
        pred_pose = self._get_RT_matrix(
            axisangle,
            translation,
            intrinsics=context["intrinsics"],
            near=context["near"],
            far=context["far"],
            image_size=context["image"].shape[-2:],
        )
        # Compute predicted relative poses (0->1 and 1->0)
        pred_01, pred_10 = split_pred_relative_two_directions(pred_pose)

        # Align GT extrinsics to view 0 for consistency with the required world frame.
        gt_pose_0to1 = None
        if "extrinsics" in context:
            extrinsics_aligned = align_world_to_view0(context["extrinsics"]) # [B, V, 4, 4]
            if extrinsics_aligned.shape[1] >= 2:
                gt_pose_0to1 = relative_pose_0_to_1(extrinsics_aligned)  # [B, 4, 4]
        # 将预测返回给上层，避免修改 batch。
        context["_enhancer_outputs"] = {
            "pred_pose_0to1": pred_01,
            "pred_pose_1to0": pred_10,
            "gt_pose_0to1": gt_pose_0to1,
        }
        return context, features

class PoseHierarchicalEnhancer(PoseSeparateEnhancer):
    def __init__(self, cfg: PoseEnhancerCfg):
        super().__init__(cfg)
    
    def _init_model(self, cfg: PoseEnhancerCfg):
        self.pose_encoder = ResnetHierarchicalEncoder(**cfg.pose_encoder)
        cfg.pose_decoder.update(num_ch_enc=self.pose_encoder.num_ch_enc)
        self.pose_decoder = PoseDecoder(**cfg.pose_decoder)
    
    def forward(
        self,
        context: dict,
        features: Sequence) -> tuple[Dict, Sequence]:
        if self.cfg.input_data == "image":
            input_data = context["image"]
        elif self.cfg.input_data == "feat":  # feat from transformer b v 128 64 64
            input_data = features[1]
        elif self.cfg.input_data == "feat_backbone":  # feat from backbone (b v) 128 32 32
            input_data = rearrange(features[0][-1], "(b v) c h w -> b v c h w", b=context["image"].shape[0])
        else:
            raise ValueError(f"Invalid input_data: {self.cfg.input_data}")
        

        if self.cfg.use_norm_xy:
            norm_xy = context["norm_xy"]
            if norm_xy.shape[-2:] != input_data.shape[-2:]:
                norm_xy = F.interpolate(norm_xy, size=input_data.shape[-2:], 
                                        mode="bilinear", align_corners=False)
            input_data_0 = torch.cat([input_data[:,0:1], norm_xy[:,0:1]], dim=2)
            input_data_1 = torch.cat([input_data[:,1:2], norm_xy[:,1:2]], dim=2)
            input_data = torch.cat([input_data_0, input_data_1], dim=1)
        input_data = rearrange(input_data, "b v c h w -> (b v) c h w")
        pose_feature = self.pose_encoder(input_data)
        axisangle, translation = self.pose_decoder(pose_feature)

        # Build SE3 with optional FOV overlap constraint
        pred_pose = self._get_RT_matrix(
            axisangle,
            translation,
            intrinsics=context["intrinsics"],
            near=context["near"],
            far=context["far"],
            image_size=context["image"].shape[-2:],
        )
        # Compute predicted relative poses (0->1 and 1->0)
        pred_01, pred_10 = split_pred_relative_two_directions(pred_pose)

        # Align GT extrinsics to view 0 for consistency with the required world frame.
        gt_pose_0to1 = None
        if "extrinsics" in context:
            extrinsics_aligned = align_world_to_view0(context["extrinsics"]) # [B, V, 4, 4]
            if extrinsics_aligned.shape[1] >= 2:
                gt_pose_0to1 = relative_pose_0_to_1(extrinsics_aligned)  # [B, 4, 4]
        # 将预测返回给上层，避免修改 batch。
        context["_enhancer_outputs"] = {
            "pred_pose_0to1": pred_01,
            "pred_pose_1to0": pred_10,
            "gt_pose_0to1": gt_pose_0to1,
        }
        return context, features

class PoseHierarchicalCCLEnhancer(PoseSeparateEnhancer):
    def __init__(self, cfg: PoseEnhancerCfg):
        super().__init__(cfg)
        self.homo_ccl = CCL_Module(ccl_mode="global",
    
    def forward(
        self,
        context: dict,
        features: Sequence) -> tuple[Dict, Sequence]:
        if self.cfg.input_data == "image":
            input_data = context["image"]
        elif self.cfg.input_data == "feat":  # feat from transformer b v 128 64 64
            input_data = features[1]
        elif self.cfg.input_data == "feat_backbone":  # feat from backbone (b v) 128 32 32
            input_data = rearrange(features[0][-1], "(b v) c h w -> b v c h w", b=context["image"].shape[0])
        else:
            raise ValueError(f"Invalid input_data: {self.cfg.input_data}")
        

        if self.cfg.use_norm_xy:
            norm_xy = context["norm_xy"]
            if norm_xy.shape[-2:] != input_data.shape[-2:]:
                norm_xy = F.interpolate(norm_xy, size=input_data.shape[-2:], 
                                        mode="bilinear", align_corners=False)
            input_data_0 = torch.cat([input_data[:,0:1], norm_xy[:,0:1]], dim=2)
            input_data_1 = torch.cat([input_data[:,1:2], norm_xy[:,1:2]], dim=2)
            input_data = torch.cat([input_data_0, input_data_1], dim=1)
        input_data = rearrange(input_data, "b v c h w -> (b v) c h w")
        pose_feature = self.pose_encoder(input_data)
        
        axisangle, translation = self.pose_decoder(pose_feature)

        # Build SE3 with optional FOV overlap constraint
        pred_pose = self._get_RT_matrix(
            axisangle,
            translation,
            intrinsics=context["intrinsics"],
            near=context["near"],
            far=context["far"],
            image_size=context["image"].shape[-2:],
        )
        # Compute predicted relative poses (0->1 and 1->0)
        pred_01, pred_10 = split_pred_relative_two_directions(pred_pose)

        # Align GT extrinsics to view 0 for consistency with the required world frame.
        gt_pose_0to1 = None
        if "extrinsics" in context:
            extrinsics_aligned = align_world_to_view0(context["extrinsics"]) # [B, V, 4, 4]
            if extrinsics_aligned.shape[1] >= 2:
                gt_pose_0to1 = relative_pose_0_to_1(extrinsics_aligned)  # [B, 4, 4]
        # 将预测返回给上层，避免修改 batch。
        context["_enhancer_outputs"] = {
            "pred_pose_0to1": pred_01,
            "pred_pose_1to0": pred_10,
            "gt_pose_0to1": gt_pose_0to1,
        }
        return context, features
 
   
class PoseXfeatEnhancer(PoseEnhancer):
    def __init__(self, cfg: PoseEnhancerCfg):
        super().__init__(cfg)
        # XFeat sparse pipeline
        self.pose_encoder = XFeatSparseEncoder(
            weights=cfg.pose_encoder.get("weights", None),
            detection_threshold=cfg.pose_encoder.get("detection_threshold", 0.0),
            top_k=cfg.pose_encoder.get("top_k", 8000),
        )
        self.pose_decoder = PoseDecoderSparse(
            cfg.pose_decoder.get("choose_closest", False))

    def forward(
        self,
        context: dict,
        features: Sequence) -> tuple[Dict, Sequence]:
        kpts0, kpts1, conf, t_scale = self.pose_encoder(context) 
        K = context["intrinsics"][:, :2]  # [B,2,3,3]

        gt_pose_0to1 = None
        if "extrinsics" in context:
            extrinsics_aligned = align_world_to_view0(context["extrinsics"])  
            if extrinsics_aligned.shape[1] >= 2:
                gt_pose_0to1 = relative_pose_0_to_1(extrinsics_aligned)
        pred = self.pose_decoder(kpts0, kpts1, conf, t_scale, K, gt_pose_0to1)  
        pred_01 = pred[:, 0]
        pred_10 = pred[:, 1]
        context["_enhancer_outputs"] = {
            "pred_pose_0to1": pred_01,
            "pred_pose_1to0": pred_10,
            "gt_pose_0to1": gt_pose_0to1,
        }
        return context, features
 