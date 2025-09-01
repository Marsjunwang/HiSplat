import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.mnn import hard_mnn, soft_mnn, soft_mnn_with_tau, topk_soft_mnn_with_tau
from ..utils.eight_point import (weighted_eight_point_single, 
    decompose_E_single, estimate_relative_pose_w8pt)
from ..utils.cam_utils import pixel_to_norm_points
from ..utils.log_optimal_transport import log_optimal_transport
from einops import rearrange
from ..utils.model import ConfidenceMLP

class PoseDecoderSparse(nn.Module):
    """
    Sparse decoder using mutual matching (hard/soft) and weighted differentiable 8-point to recover E → (R,t).
    Also predicts a non-negative scale s via a tiny MLP (optional) to scale translation.
    Inputs per batch b:
      feats0[b]: [N0,C], feats1[b]: [N1,C] (L2-normalized recommended)
      kpts0[b]:  [N0,2] pixels, kpts1[b]: [N1,2] pixels
      scores0[b]:[N0],   scores1[b]:[N1]
      K[b]:      [3,3]
    Outputs: R:[B,3,3], t_scaled:[B,3,1], s:[B]
    """

    def __init__(self, 
                 tau: float = 0.2, 
                 use_hard_mnn: bool = False, 
                 min_cossim: float = 0.0, 
                 feature_dim: int = 64,
                 ):
        super().__init__()
        self.tau = tau
        self.use_hard_mnn = use_hard_mnn
        self.min_cossim = min_cossim
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def forward(self,
                kpts0: torch.Tensor,   # [B*2,N,C] per-view descriptors
                kpts1: torch.Tensor,  # [B*2,N]
                conf: torch.Tensor,    # [B*2,N,2] pixels
                t_scale: torch.Tensor, # [B,1]
                K: torch.Tensor,      # [B,2,3,3]
                gt_pose_0to1: torch.Tensor, # [B,4,4]
                ) -> torch.Tensor:
        """
        Returns SE(3) per view direction: [B,2,4,4], where [:,0] is 0->1 and [:,1] is 1->0.
        """
        
        E, _ = estimate_relative_pose_w8pt(kpts0, kpts1, K[:,0], K[:,1], conf,
            t_scale=t_scale[:,0], choose_closest=True, T_021=gt_pose_0to1)
        pred = torch.stack([E, E.inverse()], dim=1)  # [B,2,4,4]
        return pred

