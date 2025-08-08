from pathlib import Path
from typing import Optional

import torch
from jaxtyping import Float
from torch import Tensor

from src.model.enhancer.utils.transformation_pose import (
    transformation_from_parameters,
)
from src.visualization.drawing.cameras import draw_cameras
from src.visualization.layout import hcat, vcat, add_border
from src.visualization.annotation import add_label
from src.misc.image_io import save_image


def visualize_fov_overlap(
    axisangle: Float[Tensor, "batch 1 3"],
    translation: Float[Tensor, "batch 1 3"],
    intrinsics: Float[Tensor, "batch 3 3"],
    invert: bool = False,
    epsilon_deg: float = 0.5,
    resolution: int = 512,
    save_to: Optional[str | Path] = None,
) -> Float[Tensor, "3 H W"]:
    """Visualize FOV overlap constraint before/after clamping.

    Renders two side-by-side panels:
    - Left: original relative pose
    - Right: clamped relative pose (yaw/pitch limited to FOV with epsilon)

    Returns the composed image. Optionally saves it to disk.
    """
    device = axisangle.device
    dtype = axisangle.dtype

    b = axisangle.shape[0]

    # Build original and clamped transforms.
    M_orig = transformation_from_parameters(
        axisangle, translation, invert=invert, intrinsics=intrinsics, limit_pose_to_fov_overlap=False
    )
    M_clamp = transformation_from_parameters(
        axisangle,
        translation,
        invert=invert,
        intrinsics=intrinsics,
        limit_pose_to_fov_overlap=True,
        fov_overlap_epsilon_deg=epsilon_deg,
    )

    # Camera 0 at identity (world frame == cam0).
    I = torch.eye(4, device=device, dtype=dtype).expand(b, 4, 4).clone()

    # Colors for two cameras.
    colors = torch.tensor([[1.0, 0.2, 0.2], [0.2, 0.8, 0.2]], device=device, dtype=torch.float32).expand(b, 2, 3)

    def render_panel(M: Float[Tensor, "batch 4 4"], title: str) -> Float[Tensor, "3 H W"]:
        extrinsics = torch.stack([I, M], dim=1)  # [B, 2, 4, 4]
        extrinsics = extrinsics.reshape(-1, 4, 4)
        K = intrinsics[:, None].expand(b, 2, 3, 3).reshape(-1, 3, 3)
        color = colors.reshape(-1, 3)

        panels = draw_cameras(
            resolution,
            extrinsics=extrinsics,
            intrinsics=K,
            color=color,
            near=None,
            far=None,
            margin=0.1,
            frustum_scale=0.06,
        )  # [3, 3, H, W]

        # Stack the three projections vertically.
        top, mid, bot = panels[0], panels[1], panels[2]
        image = vcat(add_border(top, 4), add_border(mid, 4), add_border(bot, 4), gap=8)
        return add_label(image, title)

    left = render_panel(M_orig, "Original Pose")
    right = render_panel(M_clamp, f"Clamped Pose (eps={epsilon_deg}°)")

    result = hcat(left, right, gap=12)

    if save_to is not None:
        save_image(result, str(save_to))

    return result

