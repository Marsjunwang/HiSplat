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
        self.conf_mlp = ConfidenceMLP(feature_dim=feature_dim, in_dim=3)

    def forward(self,
                feats: torch.Tensor,   # [B*2,N,C] per-view descriptors
                scores: torch.Tensor,  # [B*2,N]
                kpts: torch.Tensor,    # [B*2,N,2] pixels
                t_scale: torch.Tensor, # [B,1]
                K: torch.Tensor,      # [B,2,3,3]
                gt_pose_0to1: torch.Tensor, # [B,4,4]
                ) -> torch.Tensor:
        """
        Returns SE(3) per view direction: [B,2,4,4], where [:,0] is 0->1 and [:,1] is 1->0.
        """
        assert feats.dim() == 3 and scores.dim() == 2 and kpts.dim() == 3
        B2, N, C = feats.shape
        assert B2 % 2 == 0
        B = B2 // 2
        M_out = []
        
        feats = rearrange(feats, '(b v) n c -> v b n c', v=2)
        scores = rearrange(scores, '(b v) n -> v b n', v=2)
        kpts = rearrange(kpts, '(b v) n c -> v b n c', v=2)
        K = rearrange(K, 'b v c d -> v b c d')
        
        match_scores = torch.einsum('b m c, b n c -> b m n', feats[0], feats[1])
        match_scores = match_scores / C**0.5
        # TODO：use 1xN conv replace iter? mean flow?
        match_scores = log_optimal_transport(match_scores, self.bin_score, 100)  
        
        with torch.no_grad():
            M, _, _ = hard_mnn(scores=match_scores[:,:-1,:-1])
            # 如果-2列全是0,则match21是-1
            col_has_match = M.sum(dim=-2) > 0  # [B, N] - 检查每列是否有匹配
            match21 = M.argmax(dim=-2)  # [B, N]
            match21 = torch.where(col_has_match, match21, -1)  # 全零列设为-1
        
        valid = match21 >= 0
        match21_safe = match21.clamp(min=0)  
        batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, N)
        
        feats1_sel = feats[1][batch_idx, match21_safe] * valid.unsqueeze(-1)         # [B,N,C]
        kpts1_sel  = kpts[1][batch_idx, match21_safe]  * valid.unsqueeze(-1) 
        
          
        additional_input = (match_scores[batch_idx, 
            torch.arange(N, device=M.device).unsqueeze(0).expand(B, -1),
            match21_safe] * valid).unsqueeze(1)
        scores0 = scores[0].unsqueeze(1)
        scores1 = (scores[1][batch_idx, match21_safe] * valid).unsqueeze(1)
        additional_input = torch.cat(
            [scores0, scores1, additional_input], dim=1)
        conf = self.conf_mlp([
            feats[0].transpose(-2, -1), 
            feats1_sel.transpose(-2, -1), 
            additional_input]).transpose(-2, -1)
        conf = conf * (match21 >= 0).float().unsqueeze(-1)
        
        kpts0 = kpts[0]
        kpts1_exp = kpts1_sel
        
        E, _ = estimate_relative_pose_w8pt(kpts0, kpts1_exp, K[0], K[1], conf,
            t_scale=t_scale,choose_closest=False, T_021=gt_pose_0to1)
        pred = torch.stack([E, E.inverse()], dim=1)  # [B,2,4,4]
        return pred

