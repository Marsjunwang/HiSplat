import torch
import torch.nn.functional as F
from src.model.enhancer.encoder.homo_model.ccl import (
    CCL_global,
    CCL_local_tiled,
    CCL_local_tiled_vec,
)

def CCL_global_naive(c1, warp, kernel_size=3, stride=1, dilation=1,
                    padding="same"):
    """
    朴素逐样本实现，用于与分组卷积版本做数值校验。
    返回 [B, L, H, W] 的匹配体。
    """
    b, c, h, w = warp.shape
    if padding == "same":
        pad_h = ((h-1)*stride - h + dilation*(kernel_size-1) + 1) // 2
        pad_w = ((w-1)*stride - w + dilation*(kernel_size-1) + 1) // 2
    elif padding == "valid":
        pad_h = 0
        pad_w = 0
    else:
        raise ValueError(f"Invalid padding: {padding}")

    outputs = []
    for i in range(b):
        warp_i = warp[i:i+1]
        c1_i = c1[i:i+1]
        warp_i_padded = F.pad(warp_i, [pad_w, pad_w, pad_h, pad_h])
        unfolded = F.unfold(warp_i_padded,
                            kernel_size,
                            stride=stride,
                            dilation=dilation)  # [1, c*ks*ks, L]
        L = unfolded.shape[-1]
        filters = unfolded.view(1, c, kernel_size, kernel_size, L) \
                           .permute(4, 1, 2, 3, 0).squeeze(-1)  # [L, c, ks, ks]
        out_i = F.conv2d(
            c1_i,
            filters,
            bias=None,
            stride=1,
            padding=(pad_h, pad_w),
            dilation=dilation,
        )  # [1, L, H, W]
        outputs.append(out_i)
    return torch.cat(outputs, dim=0)  # [B, L, H, W]


def _make_inputs(B=2, C=8, H=16, W=16, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    c1 = torch.randn(B, C, H, W, device=device)
    warp = torch.randn(B, C, H, W, device=device)
    c1 = F.normalize(c1, p=2, dim=1)
    warp = F.normalize(warp, p=2, dim=1)
    return c1, warp


def test_global_equals_tiled_1x1():
    c1, warp = _make_inputs()
    ks, st, dil, scale = 3, 1, 1, 10
    flow_g = CCL_global(c1, warp, ks, st, dil, "same", scale)
    flow_t = CCL_local_tiled_vec(c1, warp, 1, 1, ks, st, dil, "same", scale)
    assert torch.allclose(flow_g, flow_t, rtol=1e-4, atol=1e-5), "Global should equal tiled 1x1"


def test_local_vec_equals_loop():
    # 使用 H,W 能被 tiles 整除的尺寸
    c1, warp = _make_inputs(H=24, W=20)
    ks, st, dil, scale = 3, 1, 1, 10
    Ty, Tx = 3, 5
    flow_vec = CCL_local_tiled_vec(c1, warp, Ty, Tx, ks, st, dil, "same", scale)
    flow_loop = CCL_local_tiled(c1, warp, Ty, Tx, ks, st, dil, "same", scale)
    assert torch.allclose(flow_vec, flow_loop, rtol=1e-4, atol=1e-5), "Vectorized local must match loop local"


def test_self_correlation_zero_flow():
    c1, _ = _make_inputs()
    ks, st, dil, scale = 3, 1, 1, 10
    flow = CCL_global(c1, c1, ks, st, dil, "same", scale)
    mae = flow.abs().mean().item()
    assert mae < 1e-2, f"Self-correlation flow should be near zero, got MAE={mae:.4e}"


def run_all_tests():
    tests = [
        ("global_equals_tiled_1x1", test_global_equals_tiled_1x1),
        ("local_vec_equals_loop", test_local_vec_equals_loop),
        ("self_correlation_zero_flow", test_self_correlation_zero_flow),
    ]
    passed = 0
    for name, fn in tests:
        fn()
        print(f"[PASS] {name}")
        passed += 1
    print(f"All {passed} tests passed.")
