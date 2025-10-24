import torch


def H2Mesh(H2: torch.Tensor,
           patch_size,
           grid_h: int = 8,
           grid_w: int = 8) -> torch.Tensor:
    """
    将全局单应矩阵 H2 映射为在给定 patch 上的网格顶点坐标（像素坐标系）。

    入参:
    - H2: [B, 3, 3] 的单应矩阵（批量）
    - patch_size: 标量或二元组 (H, W)，表示 patch 的像素大小
    - grid_h/grid_w: 网格在高/宽方向的单元数，输出点为 (grid_h+1)*(grid_w+1)

    返回:
    - H2_local: [B, grid_h+1, grid_w+1, 2]，每个网格点的 (x, y) 像素坐标
    """
    assert H2.dim() == 3 and H2.shape[-2:] == (3, 3), "H2 需为 [B,3,3]"

    device = H2.device
    dtype = H2.dtype
    batch_size = H2.shape[0]

    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = float(patch_size[0]), float(patch_size[1])
    else:
        patch_h = float(patch_size)
        patch_w = float(patch_size)

    step_h = torch.tensor(patch_h / grid_h, device=device, dtype=dtype)
    step_w = torch.tensor(patch_w / grid_w, device=device, dtype=dtype)

    ys = torch.arange(0, grid_h + 1, device=device, dtype=dtype) * step_h
    xs = torch.arange(0, grid_w + 1, device=device, dtype=dtype) * step_w
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [grid_h+1, grid_w+1]
    ones = torch.ones_like(xx)

    # [3, N], N = (grid_h+1)*(grid_w+1)
    ori_pts = torch.stack([xx, yy, ones], dim=0).view(3, -1)
    ori_pts = ori_pts.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 3, N]

    tar_pts = torch.bmm(H2.to(dtype), ori_pts)  # [B, 3, N]
    x_s = tar_pts[:, 0, :]
    y_s = tar_pts[:, 1, :]
    z_s = tar_pts[:, 2, :]

    # 数值稳定处理，保留符号并避免 0 除
    eps = 1e-6
    z_s = torch.where(z_s >= 0, z_s.clamp_min(eps), z_s.clamp_max(-eps))

    xy = torch.stack([x_s / z_s, y_s / z_s], dim=1)  # [B, 2, N]
    xy = xy.transpose(1, 2)  # [B, N, 2]
    H2_local = xy.view(batch_size, grid_h + 1, grid_w + 1, 2)

    return H2_local
