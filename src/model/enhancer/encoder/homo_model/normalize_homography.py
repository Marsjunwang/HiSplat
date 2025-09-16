import torch

def normalize_homography(H_pixel: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    将像素坐标系中的单应性 H 转为归一化坐标系（[-1,1]）中的 H。
    H_pixel: (B, 3, 3)
    return:  (B, 3, 3)
    """
    B = H_pixel.shape[0]
    dtype = H_pixel.dtype
    device = H_pixel.device

    s = patch_size / 2.0

    # M: norm -> pixel
    M = torch.tensor([
        [s, 0.0, s],
        [0.0, s, s],
        [0.0, 0.0, 1.0],
    ], dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)

    # M^{-1}: pixel -> norm（显式写出，避免矩阵求逆的数值与性能开销）
    Minv = torch.tensor([
        [2.0 / patch_size, 0.0, -1.0],
        [0.0, 2.0 / patch_size, -1.0],
        [0.0, 0.0, 1.0],
    ], dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)

    # H_norm = M^{-1} @ H_pixel @ M
    H_norm = torch.bmm(torch.bmm(Minv, H_pixel), M)
    return H_norm