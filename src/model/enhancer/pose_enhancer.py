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
    
    def _get_RT_matrix(self, axisangle, translation, invert=False, intrinsics=None):
        transformation = transformation_from_parameters(axisangle, translation, invert=invert, intrinsics=intrinsics)
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
        pred_pose = self._get_RT_matrix(axisangle, translation, intrinsics=context["intrinsics"])
        context["extrinsics_pred"] = pred_pose
        return context, features
    