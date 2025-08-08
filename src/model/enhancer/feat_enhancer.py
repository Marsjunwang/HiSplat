from dataclasses import dataclass
from typing import List, Sequence

from jaxtyping import Float
from torch import Tensor

from ...dataset.types import BatchedViews

from .enhancer import FeatEnhancer

@dataclass
class FeatEnhancerCfg:
    name: str
    panormic_radius: float
    panormic_resolution: List[float]
    
class PanormicFeatEnhancer(FeatEnhancer):
    def __init__(self, cfg: FeatEnhancerCfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.panormic_radius = cfg.panormic_radius
        self.panormic_resolution = cfg.panormic_resolution
        
    def forward(self, 
                context: dict,
                features: Sequence) -> tuple[dict, Sequence]:
        return {}, features