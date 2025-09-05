import torch
import torch.nn.functional as F

from .eight_point import normalize, _fallback_pose_from_kpts


def _skew_symmetric(vec: torch.Tensor) -> torch.Tensor:
    """Return skew-symmetric matrices for a batch of 3D vectors.
    vec: [..., 3] → [..., 3, 3]
    """
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    O = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([ O, -z,  y], dim=-1),
        torch.stack([ z,  O, -x], dim=-1),
        torch.stack([-y,  x,  O], dim=-1),
    ], dim=-2)
    return K


def _so3_exp(omega: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Exponential map for so(3) → SO(3).
    omega: [..., 3] → R: [..., 3, 3]
    """
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    K = _skew_symmetric(omega)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).view(*(1 for _ in omega.shape[:-1]), 3, 3)

    small = (theta < 1e-6).expand_as(theta)
    # Use series expansions for small angles
    A = torch.where(small, 1.0 - (theta**2) / 6.0 + (theta**4) / 120.0, torch.sin(theta) / (theta + eps))
    B = torch.where(small, 0.5 - (theta**2) / 24.0 + (theta**4) / 720.0, (1.0 - torch.cos(theta)) / (theta**2 + eps))

    A = A.view(*A.shape[:-1], 1, 1)
    B = B.view(*B.shape[:-1], 1, 1)
    R = I + A * K + B * (K @ K)
    return R


def _build_T(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build 4x4 homogeneous transform from rotation and translation.
    R: [...,3,3], t: [...,3]
    """
    batch_shape = R.shape[:-2]
    T = torch.zeros((*batch_shape, 4, 4), device=R.device, dtype=R.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1
    return T


def _se3_left_update(T: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Left-multiply T by exp(delta^) with small-twist approximation for translation.
    T: [B,4,4], delta: [B,6] (tx, ty, tz, wx, wy, wz)
    """
    t_delta = delta[..., :3]
    w_delta = delta[..., 3:]
    R_delta = _so3_exp(w_delta)
    T_delta = _build_T(R_delta, t_delta)
    return T_delta @ T


def _to_homogeneous_xy(xy: torch.Tensor) -> torch.Tensor:
    """xy: [B,N,2] → [B,N,3] with ones as last component."""
    ones = torch.ones_like(xy[..., :1])
    return torch.cat([xy, ones], dim=-1)


def estimate_relative_pose_se3_weighted(
    kpts0_pix: torch.Tensor,
    kpts1_pix: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    weights: torch.Tensor,
    t_scale: torch.Tensor | None,
    iters: int = 8,
    damping: float = 1e-3,
) -> torch.Tensor:
    """Estimate SE(3) from weighted 2D-2D correspondences without SVD of E/F.

    Inputs:
      - kpts0_pix: [B,N,2] pixels in view 0
      - kpts1_pix: [B,N,2] pixels in view 1
      - K0: [B,3,3] intrinsics for view 0
      - K1: [B,3,3] intrinsics for view 1
      - weights: [B,N] or [B,N,1] non-negative weights per match
      - t_scale: [B] or [B,1] desired translation magnitude per batch (optional)

    Returns: T_01 [B,4,4]
    """
    assert kpts0_pix.shape == kpts1_pix.shape and kpts0_pix.shape[-1] == 2
    B, N, _ = kpts0_pix.shape
    device = kpts0_pix.device
    dtype = kpts0_pix.dtype

    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    assert weights.shape[:2] == (B, N)
    w = torch.clamp_min(weights, 0.0)
    w_sum = w.sum(dim=1, keepdim=True) + 1e-8
    w = w / w_sum

    # Normalize keypoints to camera coordinates
    kpts0_norm = normalize(kpts0_pix, K0)
    kpts1_norm = normalize(kpts1_pix, K1)

    # Init T using a stable, differentiable fallback (direction from mean flow)
    if t_scale is None:
        t_scale = torch.ones(B, device=device, dtype=dtype)
    t_scale = t_scale.view(B) if t_scale.dim() > 0 else t_scale
    T = _fallback_pose_from_kpts(kpts0_norm, kpts1_norm, t_scale, device)

    x1 = _to_homogeneous_xy(kpts0_norm)  # [B,N,3]
    x2 = _to_homogeneous_xy(kpts1_norm)  # [B,N,3]

    I6 = torch.eye(6, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

    for _ in range(iters):
        R = T[..., :3, :3]
        t = T[..., :3, 3]

        v = (R @ x1.transpose(1, 2)).transpose(1, 2)     # [B,N,3]
        c = torch.cross(x2, v, dim=-1)                   # [B,N,3]
        r = (c * t.unsqueeze(1)).sum(dim=-1)             # [B,N]

        # Jacobians
        x2_dot_v = (x2 * v).sum(dim=-1)                  # [B,N]
        v_dot_t = (v * t.unsqueeze(1)).sum(dim=-1)       # [B,N]
        J_t = c                                          # [B,N,3]
        J_rot = x2_dot_v.unsqueeze(-1) * t.unsqueeze(1) - v_dot_t.unsqueeze(-1) * x2  # [B,N,3]
        J = torch.cat([J_t, J_rot], dim=-1)              # [B,N,6]

        # Weighted normal equations
        Jw = J * w.unsqueeze(-1)                          # [B,N,6]
        H = J.transpose(1, 2) @ Jw                        # [B,6,6]
        g = (J.transpose(1, 2) @ (w * r).unsqueeze(-1))   # [B,6,1]
        H = H + damping * I6

        # Solve H delta = -g
        try:
            delta = torch.linalg.solve(H, -g).squeeze(-1)  # [B,6]
        except RuntimeError:
            delta = torch.linalg.lstsq(H, -g).solution.squeeze(-1)

        # Update pose (left-multiply)
        T = _se3_left_update(T, delta)

    # Re-scale translation to the provided magnitude to resolve scale ambiguity
    scale_target = t_scale.view(B, 1)
    t = T[..., :3, 3]
    t_norm = torch.linalg.norm(t, dim=-1, keepdim=True).clamp_min(1e-8)
    # 避免 inplace 操作，创建新的张量
    T = T.clone()
    T[..., :3, 3] = t * (scale_target / t_norm)

    # Final sanity: if any NaNs/Infs, fall back
    if not torch.isfinite(T).all():
        T = _fallback_pose_from_kpts(kpts0_norm, kpts1_norm, t_scale, device)
    return T

