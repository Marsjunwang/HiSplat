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

    def get_batch_shim(self):
        # Perform world alignment here so enhancer remains a portable plugin.
        from .utils.pose_alignment import align_world_to_view0

        def shim(batch):
            # Align context and target extrinsics to context view-0 per-batch.
            if (
                "context" in batch
                and "extrinsics" in batch["context"]
                and batch["context"]["extrinsics"].dim() == 4
            ):
                # [B, V, 4, 4]
                context_extr = batch["context"]["extrinsics"]
                # Derive batch-wise base from each sample's first context view
                base = context_extr[:, 0]  # [B, 4, 4]
                base_inv = base.inverse().detach()[:, None]  # [B,1,4,4]
                # These are inputs; detach safe. Keep as leaves
                aligned_context = (base_inv @ batch["context"]["extrinsics"]).detach()
                batch["context"]["extrinsics"] = aligned_context
                if "target" in batch and "extrinsics" in batch["target"]:
                    aligned_target = (base_inv @ batch["target"]["extrinsics"]).detach()
                    batch["target"]["extrinsics"] = aligned_target
            return batch

        return shim
        
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