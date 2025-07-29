from typing import Optional

from .enhancer import Enhancer, PoseEnhancer, FeatEnhancer
from .pose_enhancer import PoseSeparateEnhancer, PoseEnhancerCfg
from .feat_enhancer import PanormicFeatEnhancer, FeatEnhancerCfg
from .visualization.enhancer_visualizer import EnhancerVisualizer

POSE_ENHANCERS = {
    "pose_separate": PoseSeparateEnhancer,
}

FEAT_ENHANCERS = {
    "panormic_feat": PanormicFeatEnhancer,
}

def get_pose_enhancer(cfg: PoseEnhancerCfg) -> PoseEnhancer:
    return POSE_ENHANCERS[cfg.name](cfg)

def get_feat_enhancer(cfg: FeatEnhancerCfg) -> FeatEnhancer:
    return FEAT_ENHANCERS[cfg.name](cfg) 