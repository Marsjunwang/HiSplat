import torch
import torch.nn.functional as F


def transform_torch(image_bchw: torch.Tensor, 
                    H_b33: torch.Tensor) -> torch.Tensor:
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
    x_s = (T_g[:, 0, :] / T_g[:, 2, :]).view(B, H, W)
    y_s = (T_g[:, 1, :] / T_g[:, 2, :]).view(B, H, W)

    # 组合为 grid_sample 需要的 [B, H, W, 2]，顺序为 (x, y)
    sample_grid = torch.stack([x_s, y_s], dim=-1)  # [B, H, W, 2]

    # 采样（双线性，边界复制，align_corners 对齐 linspace）
    warped = F.grid_sample(
        image_bchw,
        sample_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    return warped


if __name__ == "__main__":
    # 简单自检：单位变换应当输出等于输入
    with torch.no_grad():
        B, C, H, W = 2, 3, 32, 40
        img = torch.randn(B, C, H, W)
        H_eye = torch.eye(3).view(1, 3, 3)
        out = transform_torch(img, H_eye)
        diff = (out - img).abs().max().item()
        print("max |diff| (identity):", f"{diff:.3e}")