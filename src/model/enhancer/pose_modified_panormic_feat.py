from dataclasses import dataclass
from typing import List, Dict

from .enhancer import Enhancer, PoseEnhancer, FeatEnhancer

from torch import nn

from .factory import get_pose_enhancer, get_feat_enhancer
from .pose_enhancer import PoseEnhancerCfg
from .feat_enhancer import FeatEnhancerCfg




@dataclass
class PoseModifiedPanormicFeatCfg:
    name: str
    enhance_type: str  # "pose" or "feat" or "both"
    pose_enhancer: PoseEnhancerCfg
    feat_enhancer: FeatEnhancerCfg
    
class PoseModifiedPanormicFeatEnhancer(Enhancer):
    pose_enhancer: PoseEnhancer
    feat_enhancer: FeatEnhancer
    
    def __init__(self, cfg: PoseModifiedPanormicFeatCfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.enhance_type = cfg.enhance_type
        self.pose_enhancer = get_pose_enhancer(cfg.pose_enhancer)
        self.feat_enhancer = get_feat_enhancer(cfg.feat_enhancer)
        
    def pose_enhance(self, context, features):
        pass
    
    def feat_enhance(self, context, features):
        pass
    
    def forward(self, 
                context: Dict,
                features: List,
                ):
        pass