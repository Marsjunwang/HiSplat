import torch
import torch.nn.functional as F
from typing import Tuple


def transform_torch(image_bchw: torch.Tensor, 
                    H_b33: torch.Tensor,
                    padding_mode: str = 'border') -> torch.Tensor:
    """
    使用 PyTorch 实现与上方 TensorFlow 版本等价的透视变换采样（双线性插值）。

    入参:
    - image_bchw: [B, C, H, W] 输入图像（值域任意）
    - H_b33:      [B, 3, 3] 或 [1, 3, 3] 的单应矩阵（坐标系与 [-1,1] 归一化一致）

    返回:
    - warped:     [B, C, H, W]

    细节:
    - 采样网格使用 [-1, 1] 归一化坐标，align_corners=True（与 linspace(-1,1,N) 对齐）
    - 边界采用 border 模式（等价于 TF 中对索引的 clip）
    """
    assert image_bchw.dim() == 4, "image must be [B, C, H, W]"
    assert H_b33.dim() == 3 and H_b33.shape[-2:] == (3, 3), \
        "H must be [B,3,3] or [1,3,3]"

    B, C, H, W = image_bchw.shape
    device = image_bchw.device
    dtype = image_bchw.dtype

    if H_b33.shape[0] == 1 and B > 1:
        H_b33 = H_b33.expand(B, -1, -1)
    elif H_b33.shape[0] != B:
        raise ValueError("Batch size of H must be 1 or match image batch size")

    # 构建目标网格 [-1,1]
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]
    ones = torch.ones_like(xx)
    grid = torch.stack([xx, yy, ones], dim=0)  # [3, H, W]
    grid = grid.view(1, 3, H * W).expand(B, -1, -1)  # [B, 3, H*W]

    # 齐次坐标透视变换
    T_g = torch.bmm(H_b33.to(dtype), grid)  # [B, 3, H*W]
    denom = T_g[:, 2, :]
    # 避免分母过小引发数值不稳定，保留正负号进行对称夹取
    eps = 1e-6
    denom = torch.where(denom >= 0, denom.clamp_min(eps), denom.clamp_max(-eps))
    x_s = (T_g[:, 0, :] / denom).view(B, H, W)
    y_s = (T_g[:, 1, :] / denom).view(B, H, W)

    # 组合为 grid_sample 需要的 [B, H, W, 2]，顺序为 (x, y)
    sample_grid = torch.stack([x_s, y_s], dim=-1)  # [B, H, W, 2]

    # 采样（双线性，边界复制，align_corners 对齐 linspace）
    warped = F.grid_sample(
        image_bchw,
        sample_grid,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True,
    )
    return warped


def transform_mesh_torch(
    U: torch.Tensor,
    im_one: torch.Tensor,
    depth: torch.Tensor,
    theta: torch.Tensor,
    padding_mode: str = 'border',
):
    """
    基于网格顶点坐标 theta（[B, grid_h+1, grid_w+1, 2]，像素坐标）进行分块单应变换，
    生成整幅图像的采样网格，并对 U、im_one、depth 同步采样。

    返回: output, mask_output, warp2_depth （形状分别与输入对应，通道数保持不变）
    """
    assert U.dim() == 4 and im_one.dim() == 4 and depth.dim() == 4
    B, C, H, W = U.shape
    device = U.device
    dtype = U.dtype

    assert theta.dim() == 4 and theta.shape[-1] == 2, "theta 需为 [B, gh+1, gw+1, 2]"
    gh = theta.shape[1] - 1
    gw = theta.shape[2] - 1

    # 每块在像素坐标系下的步长（以像素中心为参考，范围 0..H-1 / 0..W-1）
    step_h = (H - 1) / gh
    step_w = (W - 1) / gw

    # 归一化坐标与像素坐标之间的变换（align_corners=True）
    M = torch.tensor(
        [[(W - 1) / 2.0, 0.0, (W - 1) / 2.0],
         [0.0, (H - 1) / 2.0, (H - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        device=device, dtype=dtype,
    )
    M_inv = torch.linalg.inv(M)

    # 生成整幅输出网格（目标采样点，按块填充）
    sample_x = torch.empty((B, H, W), device=device, dtype=dtype)
    sample_y = torch.empty((B, H, W), device=device, dtype=dtype)

    def dlt_homography_batch(pts_src: torch.Tensor, pts_dst: torch.Tensor) -> torch.Tensor:
        """根据四点对应，批量估计 H，形状：pts_* [B, 4, 2] -> H [B, 3, 3]"""
        bx = pts_src[:, :, 0]
        by = pts_src[:, :, 1]
        ux = pts_dst[:, :, 0]
        uy = pts_dst[:, :, 1]

        zeros = torch.zeros_like(bx)
        ones = torch.ones_like(bx)

        # 按 4 对应点构建 8x8 方程 A h = b，未知为 h1..h8，h9=1
        row1 = torch.stack([bx, by, ones, zeros, zeros, zeros, -bx * ux, -by * ux], dim=-1)
        row2 = torch.stack([zeros, zeros, zeros, bx, by, ones, -bx * uy, -by * uy], dim=-1)
        A = torch.stack([row1[:, 0], row2[:, 0],
                         row1[:, 1], row2[:, 1],
                         row1[:, 2], row2[:, 2],
                         row1[:, 3], row2[:, 3]], dim=1)  # [B, 8, 8]
        b = torch.stack([ux[:, 0], uy[:, 0],
                         ux[:, 1], uy[:, 1],
                         ux[:, 2], uy[:, 2],
                         ux[:, 3], uy[:, 3]], dim=1).unsqueeze(-1)  # [B,8,1]

        h = torch.linalg.solve(A.to(dtype), b.to(dtype))  # [B,8,1]
        h9 = torch.ones((B, 1, 1), device=device, dtype=dtype)
        h9el = torch.cat([h, h9], dim=1).view(B, 9)
        H = h9el.view(B, 3, 3)
        return H

    # 遍历每个小网格，估计局部 H，并将该块对应位置的 (x,y) 填入 sample grid
    for i in range(gh):
        for j in range(gw):
            # 源四角（像素坐标）
            y0 = i * step_h
            y1 = (i + 1) * step_h
            x0 = j * step_w
            x1 = (j + 1) * step_w
            src_pts = torch.tensor([
                [x0, y0], [x1, y0], [x0, y1], [x1, y1]
            ], device=device, dtype=dtype).view(1, 4, 2).expand(B, -1, -1)

            # 目标四角（来自 theta）
            dst_pts = torch.stack([
                theta[:, i, j],
                theta[:, i, j + 1],
                theta[:, i + 1, j],
                theta[:, i + 1, j + 1],
            ], dim=1).to(dtype)

            H_pix = dlt_homography_batch(src_pts, dst_pts)  # [B,3,3]
            # 变换到归一化坐标系：H_norm = M^{-1} H_pix M
            H_norm = torch.bmm(M_inv.expand(B, -1, -1), torch.bmm(H_pix, M.expand(B, -1, -1)))

            # 该块的像素范围（整型索引，包含边界）
            sh = int(round(i * (H - 1) / gh))
            eh = int(round((i + 1) * (H - 1) / gh))
            sw = int(round(j * (W - 1) / gw))
            ew = int(round((j + 1) * (W - 1) / gw))
            if i == gh - 1:
                eh = H - 1
            if j == gw - 1:
                ew = W - 1

            # 生成该块归一化网格（[-1,1]）
            ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)[sh:eh + 1]
            xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)[sw:ew + 1]
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [h_blk, w_blk]
            ones = torch.ones_like(xx)
            grid_blk = torch.stack([xx, yy, ones], dim=0).view(1, 3, -1).expand(B, -1, -1)  # [B,3,N]

            T = torch.bmm(H_norm, grid_blk)  # [B,3,N]
            z = T[:, 2, :]
            eps = 1e-6
            z = torch.where(z >= 0, z.clamp_min(eps), z.clamp_max(-eps))
            xn = (T[:, 0, :] / z).view(B, yy.shape[0], yy.shape[1])
            yn = (T[:, 1, :] / z).view(B, yy.shape[0], yy.shape[1])

            sample_x[:, sh:eh + 1, sw:ew + 1] = xn
            sample_y[:, sh:eh + 1, sw:ew + 1] = yn

    sample_grid = torch.stack([sample_x, sample_y], dim=-1)  # [B,H,W,2]

    # 采样
    output = F.grid_sample(U, sample_grid, mode='bilinear', padding_mode=padding_mode, align_corners=True)
    mask_output = F.grid_sample(im_one, sample_grid, mode='bilinear', padding_mode=padding_mode, align_corners=True)
    warp2_depth = F.grid_sample(depth, sample_grid, mode='bilinear', padding_mode=padding_mode, align_corners=True)

    return output, mask_output, warp2_depth

    
if __name__ == "__main__":
    # 简单自检：单位变换应当输出等于输入
    with torch.no_grad():
        B, C, H, W = 2, 3, 32, 40
        img = torch.randn(B, C, H, W)
        H_eye = torch.eye(3).view(1, 3, 3)
        out = transform_torch(img, H_eye)
        diff = (out - img).abs().max().item()
        print("max |diff| (identity):", f"{diff:.3e}")
