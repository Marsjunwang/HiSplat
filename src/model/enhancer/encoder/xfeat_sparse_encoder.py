import torch
from torch import nn
import torch.nn.functional as F
import os
from einops import rearrange

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

def topk_kpts_from_heatmap(heatmap: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # heatmap: [B, 1, H, W]
    B, _, H, W = heatmap.shape
    flat = heatmap.view(B, -1)                             # [B, H*W]
    k = min(k, H * W)
    vals, inds = flat.topk(k, dim=1)                      # vals: [B,k], inds: [B,k]
    ys = (inds // W).to(torch.long)                       # [B,k]
    xs = (inds %  W).to(torch.long)                       # [B,k]
    kpts = torch.stack([xs, ys], dim=-1)                  # [B,k,2], (x,y) 像素坐标，long
    return kpts, vals

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
                 top_k: int = 1024,
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
        
        self.scale_head = nn.Sequential(
            nn.Conv2d(in_channels=64*2, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, 
                      stride=4, padding=3, groups=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

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
        使用局部 soft-argmax 在每个 NMS 峰值周围做子像素细化（批量版）。
        - heatmap_logits: [B, 1, H, W]
        - kpts: [B, N, 2] (long indices, x,y)
        返回: [B, N, 2] float32 细化后的坐标
        """
        B, _, H, W = heatmap_logits.shape
        Bk, N, _ = kpts.shape
        assert B == Bk, "heatmap 和 kpts 的 batch 维不一致"

        win = self.refine_window
        half = (win - 1) / 2.0  # 以中心为原点的像素偏移

        device = heatmap_logits.device
        dtype = torch.float32

        # 生成以中心为原点的像素偏移网格（单位: 像素）
        offsets_x = torch.linspace(-half, half, steps=win, device=device, dtype=dtype)
        offsets_y = torch.linspace(-half, half, steps=win, device=device, dtype=dtype)
        yy_pix, xx_pix = torch.meshgrid(offsets_y, offsets_x, indexing='ij')  # [win,win]

        # 将像素偏移转换为 grid_sample 需要的归一化偏移 [-1,1]
        # 注意 align_corners=True 时，步长为 2/(size-1)
        xx_off = xx_pix * (2.0 / max(W - 1, 1))
        yy_off = yy_pix * (2.0 / max(H - 1, 1))
        base_grid = torch.stack([xx_off, yy_off], dim=-1).view(1, 1, win, win, 2)  # [1,1,win,win,2]

        # 计算每个关键点的中心(归一化坐标)
        kpts_f = kpts.to(dtype)
        cx = kpts_f[..., 0]  # [B,N]
        cy = kpts_f[..., 1]  # [B,N]
        cx_n = (cx * 2.0 / max(W - 1, 1)) - 1.0
        cy_n = (cy * 2.0 / max(H - 1, 1)) - 1.0
        centers = torch.stack([cx_n, cy_n], dim=-1).unsqueeze(-2).unsqueeze(-2)  # [B,N,1,1,2]

        # 构建采样网格 [B*N, win, win, 2]
        grid = (base_grid + centers).view(B * N, win, win, 2)

        # 为每个关键点重复相应 batch 的热力图
        fmap = heatmap_logits[:, :1]  # [B,1,H,W]
        fmap_rep = fmap.repeat_interleave(N, dim=0)  # [B*N,1,H,W]

        # 采样局部 patch（对边界自动零填充）
        patches = F.grid_sample(
            fmap_rep, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )  # [B*N,1,win,win]

        # 计算 soft-argmax 的概率分布（数值稳定）
        logits = (patches / self.refine_tau).view(B, N, -1)  # [B,N,win*win]
        logits = logits - logits.max(dim=-1, keepdim=True).values
        p = torch.softmax(logits, dim=-1).view(B, N, win, win)  # [B,N,win,win]

        # 以中心为原点的期望偏移（单位: 像素）
        ex = (p * xx_pix.view(1, 1, win, win)).sum(dim=(-1, -2))  # [B,N]
        ey = (p * yy_pix.view(1, 1, win, win)).sum(dim=(-1, -2))  # [B,N]

        # 锚定到中心坐标（修复: 以前中心网格却加了 x0/y0 的锚点错误）
        out = kpts_f.clone()
        out[..., 0] = cx + ex
        out[..., 1] = cy + ey

        # 对于填充的无效关键点 (x==0 且 y==0)，保持为原值
        valid = (kpts != 0).any(dim=-1)  # [B,N]
        out = torch.where(valid.unsqueeze(-1), out, kpts_f)

        return out

    def forward(self, 
                images: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, 
                           torch.Tensor, torch.Tensor]:
        # xFeat only support 2 views for matching
        assert images.dim() == 5 and images.shape[1] == 2
        b, v, c, h, w = images.shape
        x = images.reshape(b * v, c, h, w)
        feats, kpts_logits, heatmap = self.model(x)
        
        # use full image features to predict scale
        scale_feat = rearrange(feats, '(b v) c h w -> b (v c) h w', b=b, v=v)
        t_scale = self.scale_head(scale_feat)
        
        kpts_heatmap = self.get_kpts_heatmap(kpts_logits)
        
        kpts, _ = topk_kpts_from_heatmap(kpts_heatmap, self.top_k)

        # kpts = self._refine_kpts_soft(kpts_heatmap, kpts)
        
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
        
        return feats, scores, kpts, t_scale
    
