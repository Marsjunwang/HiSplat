from typing import Optional

from .enhancer import Enhancer, PoseEnhancer, FeatEnhancer
from .pose_modified_panormic_feat import (
    PoseModifiedPanormicFeatCfg,
    PoseModifiedPanormicFeatEnhancer,
)
from .visualization.enhancer_visualizer import EnhancerVisualizer

from torch import nn


ENHANCERS = {
    "pose_modified_panormic_feat": (PoseModifiedPanormicFeatEnhancer, None),
}

EnhancerCfg = PoseModifiedPanormicFeatCfg
def get_enhancer(cfg: EnhancerCfg) -> tuple[Enhancer, Optional[EnhancerVisualizer]]:
    enhancer, enhancer_visualizer = ENHANCERS[cfg.name]
    enhancer = enhancer(cfg)
    if enhancer_visualizer is not None:
        enhancer_visualizer = enhancer_visualizer(cfg.visualizer, enhancer)
    return enhancer, enhancer_visualizer