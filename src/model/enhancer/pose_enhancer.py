from .enhancer import PoseEnhancer
from dataclasses import dataclass
from typing import List, Dict

from jaxtyping import Float
from torch import Tensor

from .encoder import ResnetEncoder
from .decoder import PoseDecoder, PoseCNN

@dataclass
class PoseEnhancerCfg:
    name: str
    pose_encoder: dict
    pose_decoder: dict

class PoseSeparateEnhancer(PoseEnhancer):
    def __init__(self, cfg: PoseEnhancerCfg):
        super().__init__(self)
        self.pose_encoder = ResnetEncoder(**cfg.pose_encoder)
        cfg.pose_decoder.update(num_ch_enc=self.pose_encoder.num_ch_enc)
        self.pose_decoder = PoseDecoder(**cfg.pose_decoder)
    
    def forward(
        self, 
        context: dict, 
        features: List[Float[Tensor, "b v c h w"]]) -> Dict:
        pass
    