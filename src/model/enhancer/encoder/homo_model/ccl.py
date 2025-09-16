import torch
import torch.nn as nn
import torch.nn.functional as F


def CCL_global(c1, warp, kernel_size=3, stride=1, dilation=1, 
               padding="same", softmax_scale=10):
    """
    Global CCL
    """
    b, c, h, w = warp.shape
    if padding == "same":
        pad_h = ((h-1)*stride - h + dilation*(kernel_size-1) + 1) // 2
        pad_w = ((w-1)*stride - w + dilation*(kernel_size-1) + 1) // 2
    elif padding == "valid":
        pass
    else:
        raise ValueError(f"Invalid padding: {padding}")
    warp_padded = F.pad(warp, [pad_w, pad_w, pad_h, pad_h])
    patches_unfolded = F.unfold(warp_padded,
                                kernel_size,
                                stride=stride,
                                dilation=dilation)
    L = patches_unfolded.shape[-1]
    matching_filters = patches_unfolded.view(
        b, c, kernel_size, kernel_size, L).permute(0, 4, 1, 2, 3)

    input_reshaped = c1.view(1, b * c, h, w)
    weight_reshaped = matching_filters.reshape(
        b * L, c, kernel_size, kernel_size)

    out = F.conv2d(
        input_reshaped,
        weight_reshaped,
        bias=None,
        stride=1,
        padding=(pad_h, pad_w),
        dilation=dilation,
        groups=b,
    )
    # Reshape back to [b, L, h, w]
    match_vol = out.view(b, L, h, w)
    match_vol = F.softmax(match_vol * softmax_scale, dim=1)
    
    flow = correlation_to_flow_global(match_vol)
    return flow

def correlation_to_flow_global(match_vol: torch.Tensor) -> torch.Tensor:
    # match_vol: [B, L, H, W], L = H * W
    B, L, H, W = match_vol.shape
    device = match_vol.device
    dtype = match_vol.dtype

    # 通道索引 → 全局坐标 (iy, ix)
    idx = torch.arange(L, device=device, dtype=dtype)
    iy = (idx // W).view(1, L, 1, 1)   # [1, L, 1, 1]
    ix = (idx %  W).view(1, L, 1, 1)   # [1, L, 1, 1]

    # 当前像素坐标网格 (y, x)
    y = torch.arange(H, device=device, dtype=dtype
                     ).view(1, 1, H, 1).expand(1, 1, H, W)
    x = torch.arange(W, device=device, dtype=dtype
                     ).view(1, 1, 1, W).expand(1, 1, H, W)

    # 加权位移期望（沿 L 聚合）
    flow_h = torch.sum(match_vol * (iy - y), dim=1, keepdim=True)  # [B, 1, H, W]
    flow_w = torch.sum(match_vol * (ix - x), dim=1, keepdim=True)  # [B, 1, H, W]

    # 与 TF 一致的通道顺序: [flow_w, flow_h]
    flow = torch.cat([flow_w, flow_h], dim=1)  # [B, 2, H, W]
    return flow

def correlation_to_flow_tile(match_vol: torch.Tensor,
                             tile_origin_y: int,
                             tile_origin_x: int,
                             tile_h: int,
                             tile_w: int) -> torch.Tensor:
    """
    将局部匹配体转换为光流，坐标按照全局坐标计算位移期望。
    match_vol: [B, L_t, tile_h, tile_w], L_t = tile_h * tile_w (stride=1, same)
    返回: [B, 2, tile_h, tile_w]
    """
    B, L_t, Ht, Wt = match_vol.shape
    device = match_vol.device
    dtype = match_vol.dtype

    idx = torch.arange(L_t, device=device, dtype=dtype)
    iy_local = (idx // tile_w).view(1, L_t, 1, 1)
    ix_local = (idx %  tile_w).view(1, L_t, 1, 1)
    iy = iy_local + float(tile_origin_y)
    ix = ix_local + float(tile_origin_x)

    y = (torch.arange(Ht, device=device, dtype=dtype) + float(tile_origin_y)
         ).view(1, 1, Ht, 1).expand(1, 1, Ht, Wt)
    x = (torch.arange(Wt, device=device, dtype=dtype) + float(tile_origin_x)
         ).view(1, 1, 1, Wt).expand(1, 1, Ht, Wt)

    flow_h = torch.sum(match_vol * (iy - y), dim=1, keepdim=True)
    flow_w = torch.sum(match_vol * (ix - x), dim=1, keepdim=True)
    return torch.cat([flow_w, flow_h], dim=1)


def CCL_local_tiled(c1,
                    warp,
                    tiles_y: int = 2,
                    tiles_x: int = 2,
                    kernel_size: int = 3,
                    stride: int = 1,
                    dilation: int = 1,
                    padding: str = "same",
                    softmax_scale: float = 10):
    """
    局部(分块)版本的全局相关：将图像均匀分成 tiles_y × tiles_x 个块，
    每个块内做“全局相关”，并将结果拼接为整图光流。
    - 当 tiles_y == tiles_x == 1 时，数值上等价于 CCL_global。
    返回: [B, 2, H, W]
    """
    b, c, h, w = warp.shape

    if tiles_y <= 0 or tiles_x <= 0:
        raise ValueError("tiles_y/tiles_x must be positive")

    flow_out = torch.empty(b, 2, h, w, device=c1.device, dtype=c1.dtype)

    # 计算每个 tile 的边界（尽量均匀切分，最后一个 tile 承担余数）
    y_boundaries = [ (i * h) // tiles_y for i in range(tiles_y) ] + [h]
    x_boundaries = [ (j * w) // tiles_x for j in range(tiles_x) ] + [w]

    for ti in range(tiles_y):
        y0, y1 = y_boundaries[ti], y_boundaries[ti + 1]
        Ht = y1 - y0
        if Ht <= 0:
            continue
        for tj in range(tiles_x):
            x0, x1 = x_boundaries[tj], x_boundaries[tj + 1]
            Wt = x1 - x0
            if Wt <= 0:
                continue

            c1_tile = c1[:, :, y0:y1, x0:x1]
            warp_tile = warp[:, :, y0:y1, x0:x1]

            if padding == "same":
                pad_h = ((Ht - 1) * stride - Ht + dilation * (kernel_size - 1) + 1) // 2
                pad_w = ((Wt - 1) * stride - Wt + dilation * (kernel_size - 1) + 1) // 2
            elif padding == "valid":
                pad_h = 0
                pad_w = 0
            else:
                raise ValueError(f"Invalid padding: {padding}")

            warp_t_padded = F.pad(warp_tile, [pad_w, pad_w, pad_h, pad_h])
            patches_unfolded = F.unfold(warp_t_padded,
                                        kernel_size,
                                        stride=stride,
                                        dilation=dilation)  # [b, c*ks*ks, Lt]
            Lt = patches_unfolded.shape[-1]
            matching_filters = patches_unfolded.view(
                b, c, kernel_size, kernel_size, Lt).permute(0, 4, 1, 2, 3)

            input_reshaped = c1_tile.view(1, b * c, Ht, Wt)
            weight_reshaped = matching_filters.reshape(b * Lt, c, kernel_size, kernel_size)

            out = F.conv2d(
                input_reshaped,
                weight_reshaped,
                bias=None,
                stride=1,
                padding=(pad_h, pad_w),
                dilation=dilation,
                groups=b,
            )
            match_vol_t = out.view(b, Lt, Ht, Wt)
            match_vol_t = F.softmax(match_vol_t * softmax_scale, dim=1)

            flow_t = correlation_to_flow_tile(match_vol_t, y0, x0, Ht, Wt)
            flow_out[:, :, y0:y1, x0:x1] = flow_t

    return flow_out


def CCL_local_tiled_vec(c1,
                        warp,
                        tiles_y: int = 2,
                        tiles_x: int = 2,
                        kernel_size: int = 3,
                        stride: int = 1,
                        dilation: int = 1,
                        padding: str = "same",
                        softmax_scale: float = 10):
    """
    无 Python for 循环的向量化局部(tile) CCL。
    约束：
    - H 可被 tiles_y 整除，W 可被 tiles_x 整除；
    - stride == 1 且 padding == "same"（保证 Lt = tile_h * tile_w，坐标映射简洁）。
    条件不满足请使用 CCL_local_tiled。
    返回: [B, 2, H, W]
    """
    b, c, h, w = warp.shape
    if (h % tiles_y != 0) or (w % tiles_x != 0):
        raise ValueError("For vectorized tiles, H%tiles_y and W%tiles_x must be 0")
    if stride != 1 or padding != "same":
        raise ValueError("Vectorized tiled CCL requires stride==1 and padding=='same'")

    tile_h = h // tiles_y
    tile_w = w // tiles_x
    bt = b * tiles_y * tiles_x

    # [B, C, Ty, Th, Tx, Tw] -> [Bt, C, Th, Tw]
    c1_tiles = c1.view(b, c, tiles_y, tile_h, tiles_x, tile_w)
    c1_tiles = c1_tiles.permute(0, 2, 4, 1, 3, 5).contiguous().view(
        bt, c, tile_h, tile_w)
    warp_tiles = warp.view(b, c, tiles_y, tile_h, tiles_x, tile_w)
    warp_tiles = warp_tiles.permute(0, 2, 4, 1, 3, 5).contiguous().view(
        bt, c, tile_h, tile_w)

    # 统一 padding（same, stride=1）
    pad_h = (dilation * (kernel_size - 1)) // 2
    pad_w = (dilation * (kernel_size - 1)) // 2

    warp_t_padded = F.pad(warp_tiles, [pad_w, pad_w, pad_h, pad_h])
    patches_unfolded = F.unfold(
        warp_t_padded,
        kernel_size,
        stride=1,
        dilation=dilation,
    )  # [bt, c*ks*ks, Lt]
    Lt = patches_unfolded.shape[-1]
    # 对于 stride=1 & same，Lt == tile_h * tile_w

    matching_filters = patches_unfolded.view(bt, c, kernel_size, kernel_size, Lt)
    matching_filters = matching_filters.permute(0, 4, 1, 2, 3).contiguous()  # [bt, Lt, c, ks, ks]

    # 分组卷积同时计算所有 tiles
    input_reshaped = c1_tiles.view(1, bt * c, tile_h, tile_w)
    weight_reshaped = matching_filters.view(bt * Lt, c, kernel_size, kernel_size)
    out = F.conv2d(
        input_reshaped,
        weight_reshaped,
        bias=None,
        stride=1,
        padding=(pad_h, pad_w),
        dilation=dilation,
        groups=bt,
    )
    match_vol_t = out.view(bt, Lt, tile_h, tile_w)
    match_vol_t = F.softmax(match_vol_t * softmax_scale, dim=1)

    # 将匹配体转换为全局位移（向量化坐标）
    device = match_vol_t.device
    dtype = match_vol_t.dtype
    # 通道索引 -> tile 内坐标
    idx = torch.arange(Lt, device=device, dtype=dtype)
    iy_local = (idx // tile_w).view(1, Lt, 1, 1)  # [1, Lt, 1, 1]
    ix_local = (idx %  tile_w).view(1, Lt, 1, 1)  # [1, Lt, 1, 1]

    # 每个 tile 的原点偏移（按 Bt 展开）：
    ty = torch.arange(tiles_y, device=device, dtype=dtype)
    tx = torch.arange(tiles_x, device=device, dtype=dtype)
    ty_grid, tx_grid = torch.meshgrid(ty, tx, indexing='ij')  # [Ty, Tx]
    y0_per_tile = (ty_grid * tile_h).reshape(1, tiles_y * tiles_x)  # [1, T]
    x0_per_tile = (tx_grid * tile_w).reshape(1, tiles_y * tiles_x)  # [1, T]
    # 扩展到 Bt（为每个 batch 复制）
    y0_bt = y0_per_tile.repeat(b, 1).reshape(bt, 1, 1, 1)
    x0_bt = x0_per_tile.repeat(b, 1).reshape(bt, 1, 1, 1)

    iy = iy_local + y0_bt  # [bt, Lt, 1, 1] 广播到 [bt, Lt, Th, Tw]
    ix = ix_local + x0_bt

    # 当前像素全局坐标网格
    y_base = (torch.arange(tile_h, device=device, dtype=dtype).view(1, 1, tile_h, 1)
              + y0_bt)  # [bt, 1, Th, 1]
    x_base = (torch.arange(tile_w, device=device, dtype=dtype).view(1, 1, 1, tile_w)
              + x0_bt)  # [bt, 1, 1, Tw]

    flow_h_t = torch.sum(match_vol_t * (iy - y_base), dim=1, keepdim=True)
    flow_w_t = torch.sum(match_vol_t * (ix - x_base), dim=1, keepdim=True)
    flow_t = torch.cat([flow_w_t, flow_h_t], dim=1)  # [bt, 2, Th, Tw]

    # 还原到整图
    flow = flow_t.view(b, tiles_y, tiles_x, 2, tile_h, tile_w)
    flow = flow.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, 2, h, w)
    return flow


class CCL_Module(nn.Module):
    """
    CCL as a PyTorch module for easy integration into networks
    """
    def __init__(self, 
                 ccl_mode="global", # "global" or "local"
                 softmax_scale=10,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding="same",
                 tiles_y: int = 2,
                 tiles_x: int = 2,
                 ):
        super(CCL_Module, self).__init__()
        self.ccl_mode = ccl_mode
        self.softmax_scale = softmax_scale
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.tiles_y = tiles_y
        self.tiles_x = tiles_x
        
    def forward(self, feature1, feature2):
        """
        Args:
            feature1: [batch_size, channels, height, width] - feature map from image 1
            feature2: [batch_size, channels, height, width] - feature map from image 2
        Returns:
            flow: [batch_size, 2, height, width] - correlation flow
        """
        # TODO：1,缩小通道维度 2,sigmod是局部归一化，全局归一化靠softmax
        # Normalize features (as done in TensorFlow version)
        norm_feature1 = F.normalize(feature1, p=2, dim=1)
        norm_feature2 = F.normalize(feature2, p=2, dim=1)
        
        # Compute CCL
        if self.ccl_mode == "global":
            flow = CCL_global(norm_feature1, norm_feature2, self.kernel_size, 
                              self.stride, self.dilation, self.padding, 
                              self.softmax_scale)
        elif self.ccl_mode == "local":
            b, c, h, w = norm_feature1.shape
            can_vec = (h % self.tiles_y == 0) and (w % self.tiles_x == 0) \
                      and (self.stride == 1) and (self.padding == "same")
            if can_vec:
                flow = CCL_local_tiled_vec(norm_feature1, norm_feature2,
                                           self.tiles_y, self.tiles_x,
                                           self.kernel_size, self.stride,
                                           self.dilation, self.padding,
                                           self.softmax_scale)
            else:
                flow = CCL_local_tiled(norm_feature1, norm_feature2,
                                       self.tiles_y, self.tiles_x,
                                       self.kernel_size, self.stride,
                                       self.dilation, self.padding,
                                       self.softmax_scale)
        else:
            raise ValueError(f"Invalid ccl_mode: {self.ccl_mode}")
        
        return flow
    
if __name__ == "__main__":
    torch.manual_seed(0)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # B, C, H, W = 2, 8, 16, 16
    # ks, st, dil = 3, 1, 1
    # c1 = torch.randn(B, C, H, W, device=device)
    # warp = torch.randn(B, C, H, W, device=device)
    
    # c1 = F.normalize(c1, p=2, dim=1)
    # warp = F.normalize(warp, p=2, dim=1)

    # out_vec = CCL_global(c1, warp, kernel_size=ks, stride=st, dilation=dil, padding="same")
    # out_naive = CCL_global_naive(c1, warp, kernel_size=ks, stride=st, dilation=dil, padding="same")

    # diff = (out_vec - out_naive).abs()
    # print(f"shape: {tuple(out_vec.shape)}")
    # print(f"mean|diff|: {diff.mean().item():.6e}")
    # print(f"max |diff|: {diff.max().item():.6e}")
    # assert torch.allclose(out_vec, out_naive, rtol=1e-4, atol=1e-5), "Vectorized CCL_global mismatches naive baseline"
    # print("CCL_global vectorized vs naive: OK")
    
    # out = CCL_global(c1, c1, kernel_size=ks, stride=st, dilation=dil, padding="same")
    # B, _, H, W = c1.shape
    # yy = torch.arange(H, device=out.device).unsqueeze(1).expand(H, W)
    # xx = torch.arange(W, device=out.device).unsqueeze(0).expand(H, W)
    # l = yy * W + xx  # 对应每个 (y,x) 的通道索引

    # # 取 batch=0 的对角响应图
    # diag_map = out[0][l.reshape(-1), yy.reshape(-1), xx.reshape(-1)].view(H, W)
    # print("diag center (未归一化):", diag_map[H//2, W//2].item())          # 约 ks*ks（内点）
    # print("diag center (归一化):", (diag_map/(ks*ks))[H//2, W//2].item())   # 约 1.0

    # # 可选：统计
    # interior = diag_map[1:-1, 1:-1]
    # print("interior mean (未归一化):", interior.float().mean().item())
    # print("interior mean (归一化):", (interior/(ks*ks)).float().mean().item())

    # 运行测试入口
    try:
        from src.model.enhancer.test.ccl_test import run_all_tests
        run_all_tests()
    except Exception as e:
        print("Tests failed:", e)