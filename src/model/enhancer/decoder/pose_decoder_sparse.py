import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.mnn import hard_mnn, soft_mnn, soft_mnn_with_tau, topk_soft_mnn_with_tau
from ..utils.eight_point import weighted_eight_point_single, decompose_E_single
from ..utils.cam_utils import pixel_to_norm_points


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

    def __init__(self, tau: float = 0.2, use_hard_mnn: bool = False, min_cossim: float = 0.0, enable_scale_head: bool = True):
        super().__init__()
        self.tau = tau
        self.use_hard_mnn = use_hard_mnn
        self.min_cossim = min_cossim
        self.enable_scale_head = enable_scale_head
        if enable_scale_head:
            self.scale_head = nn.Sequential(
                nn.Linear(2, 32), nn.ReLU(inplace=True), nn.Linear(32, 1)
            )

    def _soft_mnn(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        return soft_mnn_with_tau(f0, f1, self.tau)              # mutual probabilities [N0,N1]

    def _hard_mnn(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        return hard_mnn(f0, f1)

    def forward(self,
                feats: torch.Tensor,   # [B*2,N,C] per-view descriptors
                scores: torch.Tensor,  # [B*2,N]
                kpts: torch.Tensor,    # [B*2,N,2] pixels
                K: torch.Tensor        # [B,2,3,3]
                ) -> torch.Tensor:
        """
        Returns SE(3) per view direction: [B,2,4,4], where [:,0] is 0->1 and [:,1] is 1->0.
        """
        assert feats.dim() == 3 and scores.dim() == 2 and kpts.dim() == 3
        B2, N, C = feats.shape
        assert B2 % 2 == 0
        B = B2 // 2
        M_out = []
        for b in range(B):
            f0 = feats[2 * b + 0]  # [N,C]
            f1 = feats[2 * b + 1]
            M = self._hard_mnn(f0, f1) \
                if self.use_hard_mnn else self._soft_mnn(f0, f1)  # [N,N]
            row_sum = M.sum(dim=-1, keepdim=True).clamp_min(1e-8) # [N,1]
            xy0 = kpts[2 * b + 0]                                 # [N,2]
            xy1 = kpts[2 * b + 1]
            xy1_exp = (M @ xy1) / row_sum                         # [N,2]
            w = row_sum.squeeze(-1) * scores[2 * b + 0] \
                * (M @ scores[2 * b + 1].unsqueeze(-1)).squeeze(-1)
            keep = w > 0.0
            if keep.sum() < 8:
                R = torch.eye(3, device=feats.device, dtype=feats.dtype)
                t = torch.tensor([[1.0], [0.0], [0.0]], device=feats.device, dtype=feats.dtype)
            else:
                x0n = pixel_to_norm_points(xy0[keep], K[b, 0])
                x1n = pixel_to_norm_points(xy1_exp[keep], K[b, 1])
                E = weighted_eight_point_single(x0n, x1n, w[keep])
                R, t = decompose_E_single(E)
                if self.enable_scale_head:
                    sim_avg = (M * (f0 @ f1.t())).sum() / (M.sum() + 1e-8)
                    w_avg = row_sum.mean()
                    stat = torch.stack(
                        [w_avg.squeeze(), sim_avg.squeeze()], dim=0)
                    s = F.softplus(self.scale_head(stat.unsqueeze(0))
                                   ).squeeze() + 1e-6
                    t = t * s
            # Compose [2,4,4]: 0->1 and its inverse 1->0
            Rt = torch.cat([R, t.unsqueeze(-1)], dim=-1)                                        # [3,4]
            bottom = torch.tensor([0, 0, 0, 1],
                                  device=R.device, dtype=R.dtype).view(1, 4)
            M01 = torch.cat([Rt, bottom], dim=0)                                  # [4,4]
            M10 = torch.linalg.inv(M01)
            M_out.append(torch.stack([M01, M10], dim=0))
        pred = torch.stack(M_out, dim=0)  # [B,2,4,4]
        return pred

