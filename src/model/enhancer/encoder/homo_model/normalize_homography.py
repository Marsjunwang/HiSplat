import torch

def normalize_homography(H_pixel: torch.Tensor, patch_size) -> torch.Tensor:
    """
    将像素坐标系中的单应性 H 转为归一化坐标系（[-1,1]）中的 H。
    H_pixel: (B, 3, 3)
    patch_size: int (square) 或 (H, W) 元组
    return:  (B, 3, 3)
    """
    B = H_pixel.shape[0]
    dtype = H_pixel.dtype
    device = H_pixel.device

    # 支持矩形: 统一到 (H, W)
    if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
        Hs = float(patch_size[0])
        Ws = float(patch_size[1])
    else:
        Hs = float(patch_size)
        Ws = float(patch_size)

    sy = Hs / 2.0
    sx = Ws / 2.0

    # M: norm -> pixel（注意 x 与 y 的尺度分别使用 Ws 与 Hs）
    M = torch.tensor([
        [sx, 0.0, sx],
        [0.0, sy, sy],
        [0.0, 0.0, 1.0],
    ], dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)

    # M^{-1}: pixel -> norm
    Minv = torch.tensor([
        [2.0 / Ws, 0.0, -1.0],
        [0.0, 2.0 / Hs, -1.0],
        [0.0, 0.0, 1.0],
    ], dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)

    # H_norm = M^{-1} @ H_pixel @ M
    H_norm = torch.bmm(torch.bmm(Minv, H_pixel), M)
    return H_norm