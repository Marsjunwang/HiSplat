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


class XFeatDenseEncoder(nn.Module):
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
    
