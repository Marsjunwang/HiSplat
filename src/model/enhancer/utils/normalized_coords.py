import numpy as np
import torch


def generate_normalized_xy_grid(
    K: torch.Tensor | np.ndarray,
    height: int = 256,
    width: int = 256,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Generate a per-pixel normalized image-plane coordinate grid using intrinsics K.

    The returned grid has shape [2, H, W], where the first channel is x (u-axis)
    and the second channel is y (v-axis) in normalized camera coordinates such that
    [x, y, 1]^T = inv(K) @ [u, v, 1]^T with u in [0, W-1], v in [0, H-1].

    This handles general intrinsics including skew by multiplying with inv(K).

    Args:
        K: 3x3 intrinsics as torch.Tensor or np.ndarray.
        height: Image height H.
        width: Image width W.
        device: Torch device for the returned tensor. If None, inferred from K if tensor, else 'cpu'.
        dtype: Torch dtype for the returned tensor. If None, inferred from K if tensor, else torch.float32.

    Returns:
        torch.Tensor of shape [2, H, W].
    """

    if isinstance(K, np.ndarray):
        device = torch.device(device) if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float32
        K_t = torch.as_tensor(K, dtype=dtype, device=device)
    elif isinstance(K, torch.Tensor):
        if device is None:
            device = K.device
        if dtype is None:
            dtype = K.dtype
        K_t = K.to(device=device, dtype=dtype)
    else:
        raise TypeError("K must be a torch.Tensor or a np.ndarray of shape [3,3]")

    assert K_t.shape == (3, 3), f"Expected K to be [3,3], got {tuple(K_t.shape)}"

    invK = torch.linalg.inv(K_t)

    u_coords = torch.arange(width, device=device, dtype=dtype)
    v_coords = torch.arange(height, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u_coords, v_coords, indexing="xy")  # [W,H] each, but transposed by indexing
    # Convert to [H,W]
    uu = uu.transpose(0, 1)
    vv = vv.transpose(0, 1)

    ones = torch.ones((height, width), device=device, dtype=dtype)
    pix = torch.stack((uu, vv, ones), dim=0).reshape(3, -1)  # [3, H*W]
    norm = (invK @ pix).reshape(3, height, width)  # [3, H, W]
    return norm[:2]

def to_pixel_K(K_norm, H, W):
    S = torch.tensor([[W, 0, 0],
                      [0, H, 0],
                      [0, 0, 1]], dtype=K_norm.dtype, device=K_norm.device)
    return S @ K_norm

def generate_normalized_xy_grid_batched(
    K: torch.Tensor,
    height: int,
    width: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
):
    """
    Batched variant. Supports K with leading batch/view dims, e.g. [B, V, 3, 3], [B, 3, 3], or [V, 3, 3].

    Returns a tensor with shape [..., 2, H, W] matching the leading dims of K.
    """
    assert isinstance(K, torch.Tensor), "K must be a torch.Tensor"
    if device is None:
        device = K.device
    if dtype is None:
        dtype = K.dtype

    K = to_pixel_K(K, height, width)
    invK = torch.linalg.inv(K.to(device=device, dtype=dtype))  # [..., 3, 3]

    u_coords = torch.arange(width, device=device, dtype=dtype)
    v_coords = torch.arange(height, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v_coords, u_coords, indexing="ij")  # [H,W]
    ones = torch.ones((height, width), device=device, dtype=dtype)
    pix = torch.stack((uu, vv, ones), dim=0)  # [3, H, W]

    norm = torch.einsum("...ij,jhw->...ihw", invK, pix)  # [..., 3, H, W]
    return norm[..., :2, :, :]


def add_normalized_coords_to_batch(
    batch: dict,
    key_paths: tuple[str, ...] = ("context", "target"),
    intrinsics_key: str = "intrinsics",
    image_key: str = "image",
    out_key: str = "norm_xy",
    height: int | None = None,
    width: int | None = None,
):
    """
    Compute and add normalized image-plane coordinates (2xHxW) into batch dict for given sub-keys.

    - Expects batch[part][intrinsics_key] to be a torch tensor with shape [B, V, 3, 3] or [B, 3, 3].
    - Image size is inferred from batch[part][image_key] if height/width not provided.
    - Writes batch[part][out_key] with shape [B, V, 2, H, W] (or [B, 1, 2, H, W] if V is missing).
    """
    for part in key_paths:
        if part not in batch or intrinsics_key not in batch[part]:
            continue
        K = batch[part][intrinsics_key]
        assert isinstance(K, torch.Tensor), f"{part}.{intrinsics_key} must be a torch.Tensor"

        if height is None or width is None:
            if image_key in batch[part] and isinstance(batch[part][image_key], torch.Tensor):
                # [B, V, C, H, W]
                img = batch[part][image_key]
                H_img, W_img = int(img.shape[-2]), int(img.shape[-1])
            else:
                raise ValueError("height/width must be provided when image tensor is unavailable")
        else:
            H_img, W_img = height, width

        # Ensure K has [B, V, 3, 3]
        if K.dim() == 3:
            K = K.unsqueeze(1)  # [B, 1, 3, 3]
        assert K.shape[-2:] == (3, 3), f"Expected intrinsics [...,3,3], got {tuple(K.shape)}"

        grid = generate_normalized_xy_grid_batched(K, H_img, W_img)
        batch[part][out_key] = grid  # [B, V, 2, H, W]

    return batch

def generate_normalized_xy_grid_numpy(
    K: np.ndarray,
    height: int = 256,
    width: int = 256,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Numpy variant. Returns ndarray of shape [2, H, W].

    [x, y, 1]^T = inv(K) @ [u, v, 1]^T.
    """
    K = np.asarray(K, dtype=dtype)
    assert K.shape == (3, 3), f"Expected K to be [3,3], got {K.shape}"
    invK = np.linalg.inv(K)

    u = np.arange(width, dtype=dtype)
    v = np.arange(height, dtype=dtype)
    uu, vv = np.meshgrid(u, v, indexing="xy")  # [H,W]
    ones = np.ones((height, width), dtype=dtype)
    pix = np.stack((uu, vv, ones), axis=0).reshape(3, -1)  # [3, H*W]
    norm = (invK @ pix).reshape(3, height, width)
    return norm[:2]

