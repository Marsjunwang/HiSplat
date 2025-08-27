import torch

def pixel_to_norm_points(xy_pix: torch.Tensor, 
                         K: torch.Tensor) -> torch.Tensor:
    """
    Convert pixel coords [M,2] to normalized camera coords using intrinsics 
    K [3,3].
    """
    M = xy_pix.shape[0]
    ones = torch.ones(M, 1, device=xy_pix.device, dtype=xy_pix.dtype)
    uv1 = torch.cat([xy_pix, ones], dim=-1)  # [M,3]
    invK = torch.linalg.inv(K)
    xyz = (invK @ uv1.t()).t()               # [M,3]
    return xyz[:, :2] / (xyz[:, 2:3] + 1e-8)