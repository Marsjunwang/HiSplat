from dataclasses import dataclass

@dataclass
class PoseModifiedPanormicFeatCfg:
    name: str
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int