from pathlib import Path
from typing import Optional

import torch
from jaxtyping import Float
from torch import Tensor

from src.model.enhancer.utils.transformation_pose import (
    transformation_from_parameters,
)
from .draw_cameras_colored import draw_cameras
from src.visualization.layout import hcat, vcat, add_border
from src.visualization.annotation import add_label
from src.misc.image_io import save_image


def visualize_fov_overlap(
    axisangle: Tensor,
    translation: Tensor,
    intrinsics: Tensor,
    invert: bool = False,
    epsilon_deg: float = 0.5,
    resolution: int = 512,
    save_to: Optional[str | Path] = None,
    near: Optional[Tensor | float] = None,
    far: Optional[Tensor | float] = None,
    image_size: Optional[tuple[int, int]] = None,
    batch_index: Optional[int] = 0,
    pair_mode: str = "m01",  # "m01" | "m10" | "both"
) -> Tensor:
    """Visualize FOV overlap constraint before/after clamping.

    Renders two side-by-side panels:
    - Left: original relative pose
    - Right: clamped relative pose (yaw/pitch limited to FOV with epsilon)

    Returns the composed image. Optionally saves it to disk.
    """
    device = axisangle.device
    dtype = axisangle.dtype

    # Support extra leading dims, collapse to batch N for rendering
    assert axisangle.shape[-2:] in [(1, 3), (3,)] and translation.shape[-2:] in [(1, 3), (3,)]
    assert intrinsics.shape[-2:] == (3, 3)

    # Broadcast/batch-flatten
    batch_shape = torch.broadcast_shapes(axisangle.shape[:-2] if axisangle.dim() >= 2 else axisangle.shape[:-1],
                                         translation.shape[:-2] if translation.dim() >= 2 else translation.shape[:-1],
                                         intrinsics.shape[:-2])
    def expand_to(shape, x):
        if x.shape[:-2] != shape and x.dim() >= 2:
            pad = len(shape) - len(x.shape[:-2])
            x = x.reshape((1,) * pad + x.shape)
            x = x.expand(*shape, *x.shape[-2:])
        return x
    axisangle = expand_to(batch_shape, axisangle)
    translation = expand_to(batch_shape, translation)
    intrinsics = expand_to(batch_shape, intrinsics)
    # Batch size should be the first dimension (exclude per-view dims like 2)
    b = axisangle.shape[0]

    # If requested, slice to a single batch for a clean 2-view rendering
    if batch_index is not None and b > 1:
        idx = int(max(0, min(b - 1, batch_index)))
        orig_b = b
        axisangle = axisangle[idx: idx + 1]
        translation = translation[idx: idx + 1]
        intrinsics = intrinsics[idx: idx + 1]
        # Also slice near/far if they carry batch dimension
        def slice_scalar_batch(x):
            if x is None:
                return None
            if not torch.is_tensor(x):
                return x
            t = x
            if t.dim() >= 1 and t.shape[0] == orig_b:
                return t[idx: idx + 1, ...]
            # If 1D length divisible by batch, reshape to (b, -1)
            if t.dim() == 1 and (t.numel() % orig_b == 0):
                return t.reshape(orig_b, -1)[idx: idx + 1]
            return t
        near = slice_scalar_batch(near)
        far = slice_scalar_batch(far)
        b = 1

    # Build original and clamped transforms.
    M_orig = transformation_from_parameters(
        axisangle,
        translation,
        invert=invert,
        intrinsics=intrinsics,
        near=near,
        far=far,
        image_size=image_size,
        limit_pose_to_fov_overlap=False,
    )
    M_clamp = transformation_from_parameters(
        axisangle,
        translation,
        invert=invert,
        intrinsics=intrinsics,
        near=near,
        far=far,
        image_size=image_size,
        limit_pose_to_fov_overlap=True,
        fov_overlap_epsilon_deg=epsilon_deg,
        fov_overlap_mode="finite",
    )

    # Camera 0 at identity (world frame == cam0).
    I = torch.eye(4, device=device, dtype=dtype).expand(b, 4, 4).clone()

    # Colors for two cameras.
    colors = torch.tensor([[1.0, 0.2, 0.2], [0.2, 0.8, 0.2]], device=device, dtype=torch.float32).expand(b, 2, 3)

    def render_panel(M: Tensor, title: str, pair_idx: int = 0) -> Float[Tensor, "3 H W"]:
        # Support M with shape [B,4,4] or [B,2,4,4] (view0->view1 and view1->view0)
        if M.dim() == 4 and M.shape[-3] == 2:
            Mpair = M.reshape(b, 2, 4, 4)[:, pair_idx]
        else:
            Mpair = M.reshape(b, 4, 4)
        extrinsics = torch.stack([I, Mpair], dim=1)  # [B, 2, 4, 4]
        extrinsics = extrinsics.reshape(-1, 4, 4)
        # Build per-camera intrinsics for the two views in the chosen direction
        if intrinsics.dim() == 4 and intrinsics.shape[-3] == 2:
            if pair_idx == 0:  # m01: A=view0, B=view1
                K_pair = torch.stack([intrinsics[:, 0], intrinsics[:, 1]], dim=1)  # [B,2,3,3]
            else:  # m10: A=view1, B=view0
                K_pair = torch.stack([intrinsics[:, 1], intrinsics[:, 0]], dim=1)
        else:
            K_pair = intrinsics[:, None].expand(b, 2, 3, 3)
        K = K_pair.reshape(-1, 3, 3)
        color = colors.reshape(-1, 3)

        # Align near/far shape to flattened batch (b*2) robustly
        def align_scalar(s, device, b):
            if s is None:
                return None
            s = torch.as_tensor(s, dtype=torch.float32, device=device)
            s = s.reshape(-1)
            target = b * 2
            if s.numel() == 1:
                return s.repeat(target)
            if s.numel() == b:
                return s.repeat_interleave(2)
            if s.numel() >= target:
                return s[:target]
            reps = (target + s.numel() - 1) // s.numel()
            return s.repeat(reps)[:target]

        near_flat = align_scalar(near, device, b)
        far_flat = align_scalar(far, device, b)

        panels = draw_cameras(
            resolution,
            extrinsics=extrinsics,
            intrinsics=K,
            color=color,
            near=near_flat,
            far=far_flat,
            margin=0.1,
            frustum_scale=0.06,
            near_style="solid",
            far_style="dashed",
            near_width=3,
            far_width=2,
            show_side_edges="first_only",
            other_brightness=0.35,
        )  # [3, 3, H, W]

        # Stack the three projections vertically.
        top, mid, bot = panels[0], panels[1], panels[2]
        image = vcat(add_border(top, 4), add_border(mid, 4), add_border(bot, 4), gap=8)
        return add_label(image, title)

    # Distinct colors for two views across the (now possibly sliced) batch
    colors = torch.tensor([[0.2, 0.9, 0.2], [0.9, 0.2, 0.9]], device=device, dtype=torch.float32).expand(b, 2, 3)

    def make_row(pair_idx: int, label: str):
        left_panel = render_panel(M_orig, f"Original Pose ({label})", pair_idx)
        right_panel = render_panel(M_clamp, f"Clamped Pose (eps={epsilon_deg}°) ({label})", pair_idx)
        return hcat(left_panel, right_panel, gap=12)

    rows = []
    if pair_mode == "m01":
        rows.append(make_row(0, "m01"))
    elif pair_mode == "m10":
        rows.append(make_row(1, "m10"))
    else:  # both
        rows.append(make_row(0, "m01"))
        # If M has no second pair, reuse first to avoid error
        has_second = (M_orig.dim() == 4 and M_orig.shape[-3] == 2)
        rows.append(make_row(1 if has_second else 0, "m10" if has_second else "m01"))

    # Also render a single combined panel with three colors (CamA, CamB-Orig, CamB-Clamped)
    def render_overlay_three(pair_idx: int, suffix: str) -> Float[Tensor, "3 H W"]:
        # select current (possibly sliced) single batch
        I_local = torch.eye(4, device=device, dtype=dtype).expand(1, 4, 4).clone()
        def pick_pair(M: Tensor) -> Tensor:
            if M.dim() == 4 and M.shape[-3] == 2:
                idx = min(pair_idx, 1)
                return M.reshape(b, 2, 4, 4)[0:1, idx]
            return M.reshape(b, 4, 4)[0:1]
        M_o = pick_pair(M_orig)
        M_c = pick_pair(M_clamp)
        extr = torch.cat([I_local, M_o, M_c], dim=0)  # [3,4,4]
        # Per-camera intrinsics: CamA uses source view K, CamB uses target view K (same for orig/clamped)
        if intrinsics.dim() == 4 and intrinsics.shape[-3] == 2:
            if pair_idx == 0:  # m01: A=view0, B=view1
                K_a = intrinsics.reshape(b, 2, 3, 3)[0, 0]
                K_b = intrinsics.reshape(b, 2, 3, 3)[0, 1]
            else:  # m10
                K_a = intrinsics.reshape(b, 2, 3, 3)[0, 1]
                K_b = intrinsics.reshape(b, 2, 3, 3)[0, 0]
            K_local = torch.stack([K_a, K_b, K_b], dim=0)
        else:
            K_a = intrinsics.reshape(b, 3, 3)[0]
            K_local = torch.stack([K_a, K_a, K_a], dim=0)
        colors3 = torch.tensor(
            [
                [0.2, 0.9, 0.2],  # CamA green
                [0.9, 0.2, 0.9],  # CamB original magenta
                [0.2, 0.8, 1.0],  # CamB clamped cyan
            ], device=device, dtype=torch.float32
        )
        def align_n(s, n):
            if s is None:
                return None
            if not torch.is_tensor(s):
                s = torch.tensor(s, dtype=torch.float32, device=device)
            else:
                s = s.to(device=device, dtype=torch.float32)
            if s.ndim == 0:
                return s.repeat(n)
            s_flat = s.reshape(-1)
            if s_flat.numel() < n:
                reps = (n + s_flat.numel() - 1) // s_flat.numel()
                s_flat = s_flat.repeat(reps)
            return s_flat[:n]
        near3 = align_n(near, 3)
        far3 = align_n(far, 3)
        panels = draw_cameras(
            resolution,
            extrinsics=extr,
            intrinsics=K_local,
            color=colors3,
            near=near3,
            far=far3,
            margin=0.1,
            frustum_scale=0.06,
            near_style="solid",
            far_style="dashed",
            near_width=3,
            far_width=2,
            show_side_edges="first_only",
            other_brightness=0.35,
        )
        top, mid, bot = panels[0], panels[1], panels[2]
        image = vcat(add_border(top, 4), add_border(mid, 4), add_border(bot, 4), gap=8)
        return add_label(image, f"Overlay {suffix}: A (green), B-orig (magenta), B-clamped (cyan)")

    overlays = []
    if pair_mode == "m01":
        overlays.append(render_overlay_three(0, "m01"))
    elif pair_mode == "m10":
        overlays.append(render_overlay_three(1, "m10"))
    else:
        overlays.append(render_overlay_three(0, "m01"))
        overlays.append(render_overlay_three(1, "m10"))

    result = vcat(*rows, gap=12) if len(rows) > 1 else rows[0]
    result = vcat(result, *(add_border(x, 6) for x in overlays), gap=12)

    if save_to is not None:
        save_image(result, str(save_to))

    return result


def export_fov_overlap_obj(
    axisangle: Tensor,
    translation: Tensor,
    intrinsics: Tensor,
    save_to: str | Path,
    invert: bool = False,
    epsilon_deg: float = 0.5,
    near: Optional[Tensor | float] = 0.01,
    far: Optional[Tensor | float] = 1.0,
    image_size: Optional[tuple[int, int]] = None,
    mode: str = "infinite",  # "infinite" | "finite"
) -> None:
    """Export camera FOV frustums as OBJ for CloudCompare.

    - Exports CamA (identity) frustum once, and CamB in two variants:
      original pose and clamped pose.
    - Uses `near` and `far` to form truncated pyramids.
    - Supports batched inputs; each batch is exported as a separate OBJ object.
    """
    device = axisangle.device
    dtype = axisangle.dtype

    assert axisangle.shape[-2:] in [(1, 3), (3,)] and translation.shape[-2:] in [(1, 3), (3,)]
    assert intrinsics.shape[-2:] == (3, 3)

    # Broadcast shapes like visualize_fov_overlap
    batch_shape = torch.broadcast_shapes(
        axisangle.shape[:-2] if axisangle.dim() >= 2 else axisangle.shape[:-1],
        translation.shape[:-2] if translation.dim() >= 2 else translation.shape[:-1],
        intrinsics.shape[:-2],
    )

    def expand_to(shape, x):
        if x.shape[:-2] != shape and x.dim() >= 2:
            pad = len(shape) - len(x.shape[:-2])
            x = x.reshape((1,) * pad + x.shape)
            x = x.expand(*shape, *x.shape[-2:])
        return x

    axisangle = expand_to(batch_shape, axisangle)
    translation = expand_to(batch_shape, translation)
    intrinsics = expand_to(batch_shape, intrinsics)
    B = int(torch.tensor(batch_shape).prod().item()) if len(batch_shape) > 0 else axisangle.shape[0]

    def to_scalar_or_tensor(x, like_shape):
        if x is None:
            return torch.full(like_shape, 0.01, device=device, dtype=dtype)
        if isinstance(x, (int, float)):
            return torch.full(like_shape, float(x), device=device, dtype=dtype)
        if torch.is_tensor(x):
            if x.numel() == 1:
                return torch.full(like_shape, float(x.item()), device=device, dtype=dtype)
            return x.reshape(like_shape)
        return torch.full(like_shape, 0.01, device=device, dtype=dtype)

    # Compute original and clamped transforms
    M_orig = transformation_from_parameters(
        axisangle, translation, invert=invert, intrinsics=intrinsics, near=None, far=None, image_size=image_size,
        limit_pose_to_fov_overlap=False,
    )  # [..., 4,4]
    M_clamp = transformation_from_parameters(
        axisangle, translation, invert=invert, intrinsics=intrinsics, near=None, far=None, image_size=image_size,
        limit_pose_to_fov_overlap=True, fov_overlap_epsilon_deg=epsilon_deg, fov_overlap_mode=mode,
    )  # [..., 4,4]

    # Flatten batch
    M_orig = M_orig.reshape(B, 4, 4)
    M_clamp = M_clamp.reshape(B, 4, 4)
    K = intrinsics.reshape(B, 3, 3)

    # Helpers
    def is_normalized_K(K_: torch.Tensor) -> torch.Tensor:
        cx = K_[..., 0, 2]
        cy = K_[..., 1, 2]
        return ((cx - 0.5).abs() < 0.05) & ((cy - 0.5).abs() < 0.05)

    def corner_uvs(HW: Optional[tuple[int, int]], normalized: bool) -> torch.Tensor:
        if normalized or HW is None:
            arr = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], device=device, dtype=dtype)
        else:
            H, W = float(HW[0]), float(HW[1])
            arr = torch.tensor([[0.0, 0.0], [W, 0.0], [W, H], [0.0, H]], device=device, dtype=dtype)
        return arr  # [4,2]

    def frustum_corners_cam(K_: torch.Tensor, z_near: torch.Tensor, z_far: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # K_: [3,3]; z_*: scalar tensors
        norm = bool(is_normalized_K(K_.unsqueeze(0)).item())
        uv = corner_uvs(image_size, norm)  # [4,2]
        ones = torch.ones((4, 1), device=device, dtype=dtype)
        pix = torch.cat([uv, ones], dim=-1).t()  # [3,4]
        Kinv = K_.inverse()
        d = (Kinv @ pix)  # [3,4]
        dn = (d * (z_near / (d[2:3, :] + 1e-9))).t()  # [4,3]
        df = (d * (z_far / (d[2:3, :] + 1e-9))).t()   # [4,3]
        return dn, df

    # Prepare near/far per batch
    near_b = to_scalar_or_tensor(near, (B,))
    far_b = to_scalar_or_tensor(far, (B,))

    # Build OBJ
    save_to = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# FOV frustums exported for CloudCompare")

    v_offset = 0

    def add_frustum_obj(name: str, verts_near: torch.Tensor, verts_far: torch.Tensor):
        nonlocal v_offset
        lines.append(f"o {name}")
        # vertices
        for v in torch.cat([verts_near, verts_far], dim=0):
            lines.append(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}")
        # indices (OBJ 1-based)
        n0, n1, n2, n3 = v_offset + 1, v_offset + 2, v_offset + 3, v_offset + 4
        f0, f1, f2, f3 = v_offset + 5, v_offset + 6, v_offset + 7, v_offset + 8
        # near and far faces
        lines.append(f"f {n0} {n1} {n2}")
        lines.append(f"f {n0} {n2} {n3}")
        lines.append(f"f {f0} {f1} {f2}")
        lines.append(f"f {f0} {f2} {f3}")
        # sides (as two triangles per quad)
        lines.append(f"f {n0} {n1} {f1}")
        lines.append(f"f {n0} {f1} {f0}")
        lines.append(f"f {n1} {n2} {f2}")
        lines.append(f"f {n1} {f2} {f1}")
        lines.append(f"f {n2} {n3} {f3}")
        lines.append(f"f {n2} {f3} {f2}")
        lines.append(f"f {n3} {n0} {f0}")
        lines.append(f"f {n3} {f0} {f3}")
        v_offset += 8

    for i in range(B):
        Ki = K[i]
        z_near = near_b[i]
        z_far = far_b[i]
        # Cam A (identity)
        dn_a, df_a = frustum_corners_cam(Ki, z_near, z_far)  # [4,3] each
        # Transform to world with identity -> unchanged
        add_frustum_obj(f"batch{i}_camA", dn_a, df_a)

        # Cam B original
        M = M_orig[i]
        R = M[:3, :3]
        t = M[:3, 3]
        dn_b, df_b = frustum_corners_cam(Ki, z_near, z_far)
        dn_bw = (R @ dn_b.t()).t() + t
        df_bw = (R @ df_b.t()).t() + t
        add_frustum_obj(f"batch{i}_camB_orig", dn_bw, df_bw)

        # Cam B clamped
        Mc = M_clamp[i]
        Rc = Mc[:3, :3]
        tc = Mc[:3, 3]
        dn_bc, df_bc = frustum_corners_cam(Ki, z_near, z_far)
        dn_bcw = (Rc @ dn_bc.t()).t() + tc
        df_bcw = (Rc @ df_bc.t()).t() + tc
        add_frustum_obj(f"batch{i}_camB_clamp_{mode}", dn_bcw, df_bcw)

    save_to.write_text("\n".join(lines))
