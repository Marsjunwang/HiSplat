from dataclasses import dataclass
from typing import List, Dict, Sequence

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
        context, features = self.pose_enhancer(context, features)
        return context, features
    
    def feat_enhance(self, context, features):
        context, features = self.feat_enhancer(context, features)
        return context, features
    
    def forward(self, 
                context,
                features,
                ) -> tuple[dict, Sequence]:
        if self.enhance_type == "pose":
            context, features = self.pose_enhance(context, features)
        elif self.enhance_type == "feat":
            context, features = self.feat_enhance(context, features)
        elif self.enhance_type == "both":
            context, features = self.pose_enhance(context, features)
            context, features = self.feat_enhance(context, features)
        else:
            raise ValueError(f"Invalid enhance_type: {self.enhance_type}")
        return context, features