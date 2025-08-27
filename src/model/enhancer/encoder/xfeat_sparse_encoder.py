import torch
from torch import nn
import torch.nn.functional as F
import os

try:
    # 尝试相对导入（正常项目运行）
    from .xFeat.model import XFeatModel
    from .xFeat.interpolator import InterpolateSparse2d
except ImportError:
    # 回退到通过import_utils导入（独立运行时）
    import sys
    from pathlib import Path
    xfeat_dir = Path(__file__).parent / 'xFeat'
    sys.path.insert(0, str(xfeat_dir))
    try:
        from import_utils import get_xfeat_model, get_interpolate_sparse2d
        XFeatModel = get_xfeat_model()
        InterpolateSparse2d = get_interpolate_sparse2d()
    except ImportError:
        # 最后的备选方案：直接导入
        from model import XFeatModel
        from interpolator import InterpolateSparse2d
import cv2
import numpy as np

import torch
import torch.nn.functional as F

def spatial_softargmax2d(logits: torch.Tensor, tau: float = 0.2, return_cov: bool = False):
    """
    logits: [B, C, H, W] 或 [B, 1, H, W] 的热力图（未归一化）
    tau: 温度系数；越小越“硬”
    return_cov: 是否返回协方差（不确定度估计）

    返回:
      coords: [B, C, 2]，每通道一个 (x, y) 像素坐标
      (可选) cov: [B, C, 2, 2]，二阶矩减去均值构成的协方差
    """
    assert logits.dim() == 4
    B, C, H, W = logits.shape
    device = logits.device
    dtype = logits.dtype

    # 数值稳定的 softmax：减去每通道最大值
    l = (logits / tau).reshape(B, C, -1)
    l = l - l.max(dim=-1, keepdim=True).values
    p = F.softmax(l, dim=-1).reshape(B, C, H, W)  # [B,C,H,W]

    # 构建网格（像素坐标）
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")   # [H,W]
    xx = xx.expand(B, C, H, W)
    yy = yy.expand(B, C, H, W)

    # 期望坐标
    ex = (p * xx).sum(dim=(-1, -2))  # [B,C]
    ey = (p * yy).sum(dim=(-1, -2))  # [B,C]
    coords = torch.stack([ex, ey], dim=-1)  # [B,C,2]  (x,y)

    if not return_cov:
        return coords

    # 协方差（可作为不确定度）：E[(x-Ex)(x-Ex)^T]
    ex2 = (p * (xx - ex.unsqueeze(-1).unsqueeze(-1))**2).sum(dim=(-1, -2))
    ey2 = (p * (yy - ey.unsqueeze(-1).unsqueeze(-1))**2).sum(dim=(-1, -2))
    exy = (p * (xx - ex.unsqueeze(-1).unsqueeze(-1)) * (yy - ey.unsqueeze(-1).unsqueeze(-1))).sum(dim=(-1, -2))
    cov = torch.zeros(B, C, 2, 2, device=device, dtype=dtype)
    cov[..., 0, 0] = ex2
    cov[..., 1, 1] = ey2
    cov[..., 0, 1] = cov[..., 1, 0] = exy
    return coords, cov

class XFeatSparseEncoder(nn.Module):
    """
    Wraps XFeatModel to expose dense or sparse keypoints, features and heatmaps
    for two-view pose.
    Returns feats [Bx2,...] and heatmaps [Bx2,...].
    """

    def __init__(self, 
                 weights: str | None = os.path.abspath(
                     os.path.dirname(__file__)) + '/xFeat/weights/xfeat.pt',
                 detection_threshold: float = 0.0,
                 top_k: int = -1,
                 refine_window: int = 5,
                 refine_tau: float = 0.2) -> None:
        super().__init__()
        self.model = XFeatModel()
        if weights is not None:
            print(f"Loading weights from {weights}")
            self.model.load_state_dict(torch.load(weights))
        
        self.detection_threshold = detection_threshold
        self._nearest = InterpolateSparse2d('nearest')
        self._bilinear = InterpolateSparse2d('bilinear')
        self.interpolator = InterpolateSparse2d('bicubic')
        self.top_k = top_k
        self.refine_window = refine_window
        self.refine_tau = refine_tau

    def get_kpts_heatmap(self, kpts, softmax_temp = 1.0):
        scores = F.softmax(kpts*softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
        return heatmap

    def NMS(self, x, threshold = 0.05, kernel_size = 5):
        B, _, H, W = x.shape
        pad=kernel_size//2
        local_max = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(x) for x in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

		#Pad kpts and build (B, N, 2) tensor
        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b]

        return pos

    def _refine_kpts_soft(self, 
                          heatmap_logits: torch.Tensor, 
                          kpts: torch.Tensor) -> torch.Tensor:
        """
        Refine integer keypoints with local soft-argmax around each NMS peak.
        heatmap_logits: [B, 1, H, W]
        kpts: [B, N, 2] (long indices, x,y)
        returns: [B, N, 2] float32 refined coordinates
        """
        B, _, H, W = heatmap_logits.shape
        win = self.refine_window
        half = win // 2
        out = kpts.detach().clone().to(torch.float32)
        for b in range(B):
            for n in range(kpts.shape[1]):
                xy = kpts[b, n]
                # skip padded zeros
                if (xy == 0).all():
                    continue
                x = int(xy[0].item())
                y = int(xy[1].item())
                x0 = max(0, x - half); x1 = min(W, x + half + 1)
                y0 = max(0, y - half); y1 = min(H, y + half + 1)
                patch = heatmap_logits[b, 0, y0:y1, x0:x1]
                # pad to fixed window size if near borders
                ph, pw = patch.shape
                if ph != win or pw != win:
                    pad = (0, win - pw, 0, win - ph)
                    patch = F.pad(patch, pad, mode='constant', value=0)
                logits = (patch / self.refine_tau).reshape(-1)
                logits = logits - logits.max()
                p = torch.softmax(logits, dim=0).reshape(win, win)
                yy, xx = torch.meshgrid(
                    torch.arange(win, 
                                 device=patch.device, 
                                 dtype=torch.float32),
                    torch.arange(win, 
                                 device=patch.device, 
                                 dtype=torch.float32), 
                    indexing='ij')
                # Center the grid around the keypoint
                yy = yy - half
                xx = xx - half
                ex = (p * xx).sum()
                ey = (p * yy).sum()
                out[b, n, 0] = x0 + ex
                out[b, n, 1] = y0 + ey
        return out

    def forward(self, 
                images: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # xFeat only support 2 views for matching
        assert images.dim() == 5 and images.shape[1] == 2
        b, v, c, h, w = images.shape
        x = images.reshape(b * v, c, h, w)
        feats, kpts_logits, heatmap = self.model(x)
        kpts_heatmap = self.get_kpts_heatmap(kpts_logits)
        kpts = self.NMS(
            kpts_heatmap, threshold=self.detection_threshold, kernel_size=5
            )
        # # Provide gradient to kpts
        kpts = self._refine_kpts_soft(kpts_heatmap, kpts)
        
        scores = (
            self._nearest(kpts_heatmap, kpts, h, w) * 
            self._bilinear(heatmap, kpts, h, w)
            ).squeeze(-1)
        scores[torch.all(kpts == 0, dim=-1)] = -1
        #Select top-k features
        idxs = torch.argsort(-scores)
        kpts_x  = torch.gather(kpts[...,0], -1, idxs)
        kpts_y  = torch.gather(kpts[...,1], -1, idxs)
        kpts = torch.cat([kpts_x[...,None], kpts_y[...,None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)
        if self.top_k > 0:
            kpts = kpts[:, :self.top_k]
            scores = scores[:, :self.top_k]
        

        kpts = kpts.to(torch.float32)
        feats = self.interpolator(feats, kpts, h, w)
        feats = F.normalize(feats, dim=-1)
        
        return feats, scores, kpts
    
