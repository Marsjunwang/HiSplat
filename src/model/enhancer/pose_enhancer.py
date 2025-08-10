from numpy import AxisError
from .enhancer import PoseEnhancer
from dataclasses import dataclass
from typing import List, Dict, Sequence

from jaxtyping import Float
from torch import Tensor

from .encoder import ResnetEncoder
from .decoder import PoseDecoder, PoseCNN
from einops import rearrange
from .utils.transformation_pose import transformation_from_parameters

@dataclass
class PoseEnhancerCfg:
    name: str
    input_data: str
    limit_pose_to_fov_overlap: bool
    fov_overlap_epsilon_deg: float
    pose_encoder: dict
    pose_decoder: dict

class PoseSeparateEnhancer(PoseEnhancer):
    def __init__(self, cfg: PoseEnhancerCfg):
        super().__init__(cfg)
        self.pose_encoder = ResnetEncoder(**cfg.pose_encoder)
        cfg.pose_decoder.update(num_ch_enc=self.pose_encoder.num_ch_enc)
        self.pose_decoder = PoseDecoder(**cfg.pose_decoder)
        if self.cfg.input_data == "image":
            assert self.pose_encoder.channels == 3
        elif self.cfg.input_data == "feat":
            assert self.pose_encoder.channels == 128
            
        self.limit_pose_to_fov_overlap = cfg.limit_pose_to_fov_overlap
        self.fov_overlap_epsilon_deg = cfg.fov_overlap_epsilon_deg
    
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
        elif self.cfg.input_data == "feat":
            input_data = rearrange(features[1], "b v c h w -> b (v c) h w")
        else:
            raise ValueError(f"Invalid input_data: {self.cfg.input_data}")
        
        pose_feature = self.pose_encoder(input_data)
        axisangle, translation = self.pose_decoder(pose_feature)
        pred_pose = self._get_RT_matrix(axisangle, 
                                        translation, 
                                        intrinsics=context["intrinsics"],
                                        near=context["near"],
                                        far=context["far"],
                                        image_size=context["image"].shape[-2:])
        context["extrinsics_pred"] = pred_pose
        return context, features
    