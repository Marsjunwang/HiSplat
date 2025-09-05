import torch
from torch import nn
import torch.nn.functional as F
import os
from einops import rearrange
from .scale_head import ScaleHead
from ..utils.mnn import hard_mnn
from ..utils.model import ConfidenceMLP
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
                 refine_tau: float = 0.2,
                 feat_size: tuple[int, int] = (32, 32)) -> None:
        super().__init__()
        self.model = XFeatModel()
        if weights is not None:
            print(f"Loading weights from {weights}")
            # help me finish
            weights = torch.load(weights, map_location='cpu')
            # Remove keys related to fine_matcher to avoid loading errors
            keys_to_remove = [k for k in list(weights.keys()) if 'fine_matcher' in k]
            for k in keys_to_remove:
                del weights[k]
            # Some checkpoints may have 'module.' prefix, strip if needed
            state_dict = {}
            for k, v in weights.items():
                if k.startswith('module.'):
                    state_dict[k[len('module.'):]] = v
                else:
                    state_dict[k] = v
            self.model.load_state_dict(state_dict, strict=False)
        
        self.detection_threshold = detection_threshold
        self._nearest = InterpolateSparse2d('nearest')
        self._bilinear = InterpolateSparse2d('bilinear')
        self.interpolator = InterpolateSparse2d('bicubic')
        self.top_k = top_k
        self.refine_window = refine_window
        self.refine_tau = refine_tau
        
        self.scale_head = ScaleHead()
        self.conf_mlp = ConfidenceMLP(feature_dim=129, in_dim=3)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mesh_xy = self.create_xy(feat_size[0], feat_size[1], device)

    def get_kpts_heatmap(self, kpts, softmax_temp = 1.0):
        scores = F.softmax(kpts*softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
        return heatmap

    def create_xy(self, h, w, dev):
        y, x = torch.meshgrid(torch.arange(h, device = dev), 
								torch.arange(w, device = dev), indexing='ij')
        xy = torch.cat([x[..., None],y[..., None]], -1).reshape(-1,2)
        return xy
    
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

    def matches(self, feats, kpts, scores, K):
        assert feats.dim() == 3 and scores.dim() == 2 and kpts.dim() == 3
        B2, N, C = feats.shape
        assert B2 % 2 == 0
        B = B2 // 2
        
        feats = rearrange(feats, '(b v) n c -> v b n c', v=2)
        scores = rearrange(scores, '(b v) n -> v b n', v=2)
        kpts = rearrange(kpts, '(b v) n c -> v b n c', v=2)
        K = rearrange(K, 'b v c d -> v b c d')
        
        match_scores = torch.einsum('b m c, b n c -> b m n', feats[0], feats[1])
        match_scores = match_scores / C**0.5
        # TODO：use 1xN conv replace iter? mean flow?
        # match_scores = log_optimal_transport(match_scores, self.bin_score, 100)  
        
        with torch.no_grad():
            _, match12, match21 = hard_mnn(scores=match_scores)
        # match21 = torch.argmax(match_scores.permute(0,2,1), dim=-1)
            
        valid = match21 >= 0
        batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, N)
        
        feats1_sel = feats[1][batch_idx, match21] * valid.unsqueeze(-1)         # [B,N,C]
        kpts1_sel  = kpts[1][batch_idx, match21]  * valid.unsqueeze(-1) 
        
          
        additional_input = (match_scores[batch_idx, 
            torch.arange(N, device=match21.device).unsqueeze(0).expand(B, -1),
            match21] * valid).unsqueeze(1)
        scores0 = scores[0].unsqueeze(1)
        scores1 = (scores[1][batch_idx, match21] * valid).unsqueeze(1)
        additional_input = torch.cat(
            [scores0, scores1, additional_input], dim=1)
        conf = self.conf_mlp([
            feats[0].transpose(-2, -1), 
            feats1_sel.transpose(-2, -1), 
            additional_input]).transpose(-2, -1)
        conf = conf * (match21 >= 0).float().unsqueeze(-1)
        
        kpts0 = kpts[0]
        kpts1 = kpts1_sel.to(torch.float32)
        
        offsets = self.model.fine_matcher(
            torch.cat([feats[0], feats1_sel],dim=-1).permute(0,2,1))
        # Conv1d输出形状是 [B, 64, N]，需要重塑为 [B, N, 8, 8]
        offsets = offsets.permute(0, 2, 1)  # [B, N, 64]
        offsets = offsets.view(B, N, 8, 8)  # [B, N, 8, 8]
        offsets = self.subpix_softmax2d(offsets)
        kpts0 = offsets + kpts0
        
        return kpts0, kpts1, conf 

    def subpix_softmax2d(self, heatmaps, temp = 3):
        """
        计算亚像素精度的偏移量
        Args:
            heatmaps: [B, N, H, W] 热力图
        Returns:
            coords: [B, N, 2] 亚像素偏移量
        """
        B, N, H, W = heatmaps.shape
        heatmaps = torch.softmax(
            temp * heatmaps.view(B, N, H*W), -1).view(B, N, H, W)
        
        # 生成坐标网格
        x, y = torch.meshgrid(torch.arange(W, device=heatmaps.device), 
                             torch.arange(H, device=heatmaps.device), 
                             indexing='xy')
        x = x - (W//2)  # 以中心为原点
        y = y - (H//2)
        
        # 计算期望偏移
        coords_x = (x[None, None, ...] * heatmaps).sum(dim=(-1, -2))  # [B, N]
        coords_y = (y[None, None, ...] * heatmaps).sum(dim=(-1, -2))  # [B, N]
        
        coords = torch.stack([coords_x, coords_y], dim=-1)  # [B, N, 2]
        return coords
        
    def forward(self, 
                context: dict
                ) -> tuple[torch.Tensor, torch.Tensor, 
                           torch.Tensor, torch.Tensor]:
        # xFeat only support 2 views for matching
        images = context["image"]
        device = images.device
        assert images.dim() == 5 and images.shape[1] == 2
        b, v, c, h, w = images.shape
        x = images.reshape(b * v, c, h, w)
        feats, kpts_logits, heatmap = self.model(x)
        full_feats = torch.cat([feats, kpts_logits], dim=1)
        _, feat_channels, f_h, f_w = full_feats.shape
        scale = torch.as_tensor([f_w, f_h], device=device).view(1, 1, 2, 1)
        context["intrinsics"][:, :, :2] *= scale
        
        scale_feat = rearrange(full_feats, 
                               '(b v) c h w -> b (v c) h w', b=b, v=v)
        t_scale = self.scale_head(scale_feat)
        
        heatmap = heatmap.permute(0,2,3,1).reshape(b * v, -1)
        full_feats = full_feats.permute(0,2,3,1).reshape(
            b * v, -1, feat_channels)
        kpts = self.mesh_xy.to(images.device).expand(b * v, -1, -1)
        _, top_k = torch.topk(heatmap, 
                              min(self.top_k, heatmap.shape[-1]), dim=-1)

        mkpts = torch.gather(kpts, 1, top_k[..., None].expand(-1, -1, 2))
        mfeats = torch.gather(full_feats, 1, 
                              top_k[..., None].expand(-1, -1, feat_channels))
        scores = torch.gather(heatmap, 1, top_k)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return *self.matches(mfeats, mkpts, scores, context["intrinsics"]), t_scale
    
