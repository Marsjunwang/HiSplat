from typing import Optional

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from src.visualization.drawing.lines import draw_lines
from src.visualization.drawing.types import Scalar, sanitize_scalar
from src.visualization.annotation import add_label


@torch.no_grad()
def draw_cameras(
    resolution: int,
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    color: Float[Tensor, "batch 3"],
    near: Optional[Scalar] = None,
    far: Optional[Scalar] = None,
    margin: float = 0.1,
    frustum_scale: float = 0.05,
    # Style controls for readability
    near_style: str = "solid",      # "solid" | "dashed"
    far_style: str = "dashed",       # "solid" | "dashed"
    near_width: int = 3,
    far_width: int = 2,
    show_side_edges: str = "first_only",  # "all" | "first_only"
    other_brightness: float = 0.45,  # dim non-focused cameras
) -> Float[Tensor, "3 3 height width"]:
    """Draw colored camera frustums with colored near/far rectangles.

    - Colors apply to both frustum sides and near/far plane rectangles per camera.
    - Works with batched inputs. Returns three stacked projections (YZ, ZX, XY).
    """
    device = extrinsics.device

    minima, maxima = compute_aabb(extrinsics, intrinsics, near, far)
    scene_minima, scene_maxima = compute_equal_aabb_with_margin(minima, maxima, margin=margin)
    span = (scene_maxima - scene_minima).max()

    corner_depth = (span * frustum_scale)[None]
    frustum_corners = unproject_frustum_corners(extrinsics, intrinsics, corner_depth)
    have_near = near is not None
    have_far = far is not None
    if have_near:
        near_depth = sanitize_scalar(near, device)
        near_corners = unproject_frustum_corners(extrinsics, intrinsics, near_depth)
    if have_far:
        far_depth = sanitize_scalar(far, device)
        far_corners = unproject_frustum_corners(extrinsics, intrinsics, far_depth)

    projections = []
    for projected_axis in range(3):
        image = torch.zeros((3, resolution, resolution), dtype=torch.float32, device=device)
        x_axis = (projected_axis + 1) % 3
        y_axis = (projected_axis + 2) % 3

        def project(points: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 2"]:
            x = points[..., x_axis]
            y = points[..., y_axis]
            return torch.stack([x, y], dim=-1)

        x_range, y_range = torch.stack((project(scene_minima), project(scene_maxima)), dim=-1)

        # Frustum sides (center to corners). Optionally dim non-focused cameras.
        origins = project(extrinsics[:, :3, 3])  # [B, 2]
        corners = project(frustum_corners)       # [B, 4, 2]
        side_color = color.clone()
        if show_side_edges == "first_only" and side_color.shape[0] > 1:
            side_color[1:] = side_color[1:] * other_brightness
        start = [repeat(origins, "b xy -> (b p) xy", p=4), rearrange(corners.roll(1, 1), "b p xy -> (b p) xy")]
        start = rearrange(torch.cat(start, dim=0), "(r b p) xy -> (b r p) xy", r=2, p=4)
        image = draw_lines(
            image,
            start,
            repeat(corners, "b p xy -> (b r p) xy", r=2),
            color=repeat(side_color, "b c -> (b r p) c", r=2, p=4),
            width=2,
            x_range=x_range,
            y_range=y_range,
        )

        # Colored near/far rectangles
        def _draw_polyline_rect(proj: Tensor, width: int, style: str) -> Tensor:
            # proj: [B, 4, 2], edges between consecutive corners
            if style == "solid":
                return draw_lines(
                    image,
                    rearrange(proj, "b p xy -> (b p) xy"),
                    rearrange(proj.roll(1, 1), "b p xy -> (b p) xy"),
                    color=repeat(color, "b c -> (b p) c", p=4),
                    width=width,
                    x_range=x_range,
                    y_range=y_range,
                )
            # dashed: split each edge into segments and draw every other segment
            nseg = 16
            p0 = proj
            p1 = proj.roll(1, 1)
            t = torch.linspace(0, 1, nseg + 1, device=device, dtype=torch.float32)
            t0 = t[:-1][None, None, :, None]  # [1,1,nseg,1]
            t1 = t[1:][None, None, :, None]
            seg_start = (1 - t0) * p0[..., None, :] + t0 * p1[..., None, :]
            seg_end = (1 - t1) * p0[..., None, :] + t1 * p1[..., None, :]
            # keep odd segments only
            keep = torch.arange(nseg, device=device) % 2 == 1
            seg_start = seg_start[..., keep, :]
            seg_end = seg_end[..., keep, :]
            # reshape to lines
            B = proj.shape[0]
            s_kept = int(keep.sum().item())
            seg_start = rearrange(seg_start, "b p s xy -> (b p s) xy")  # [(B*4*s) , 2]
            seg_end = rearrange(seg_end, "b p s xy -> (b p s) xy")
            # expand per-camera colors to lines
            color_lines = (
                color[:, None, None, :]  # [B,1,1,3]
                .expand(B, 4, s_kept, 3)
                .reshape(B * 4 * s_kept, 3)
            )
            return draw_lines(
                image,
                seg_start,
                seg_end,
                color=color_lines,
                width=width,
                x_range=x_range,
                y_range=y_range,
            )

        def draw_rect(corners_xyz: Tensor, width: int, style: str) -> Tensor:
            proj = project(corners_xyz)  # [B, 4, 2]
            return _draw_polyline_rect(proj, width=width, style=style)

        if have_near:
            image = draw_rect(near_corners, width=near_width, style=near_style)
        if have_far:
            image = draw_rect(far_corners, width=far_width, style=far_style)
        if have_near and have_far:
            # Connect near-far corresponding corners (optional, colored)
            n_proj = project(near_corners)
            f_proj = project(far_corners)
            image = draw_lines(
                image,
                rearrange(n_proj, "b p xy -> (b p) xy"),
                rearrange(f_proj, "b p xy -> (b p) xy"),
                color=repeat(color, "b c -> (b p) c", p=4),
                width=1,
                x_range=x_range,
                y_range=y_range,
            )

        label = f"{'XYZ'[x_axis]}{'XYZ'[y_axis]} Projection"
        image = add_label(image, label)
        projections.append(image)

    return torch.stack(projections)


def compute_aabb(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Optional[Scalar] = None,
    far: Optional[Scalar] = None,
) -> tuple[Float[Tensor, "3"], Float[Tensor, "3"]]:
    device = extrinsics.device
    points = [extrinsics[:, :3, 3]]
    if near is not None:
        z = sanitize_scalar(near, device)
        corners = unproject_frustum_corners(extrinsics, intrinsics, z)
        points.append(rearrange(corners, "b p xyz -> (b p) xyz"))
    if far is not None:
        z = sanitize_scalar(far, device)
        corners = unproject_frustum_corners(extrinsics, intrinsics, z)
        points.append(rearrange(corners, "b p xyz -> (b p) xyz"))
    pts = torch.cat(points, dim=0)
    return pts.min(dim=0).values, pts.max(dim=0).values


def compute_equal_aabb_with_margin(
    minima: Float[Tensor, "3"],
    maxima: Float[Tensor, "3"],
    margin: float = 0.1,
) -> tuple[Float[Tensor, "3"], Float[Tensor, "3"]]:
    midpoint = (maxima + minima) * 0.5
    span = (maxima - minima).max() * (1 + margin)
    scene_minima = midpoint - 0.5 * span
    scene_maxima = midpoint + 0.5 * span
    return scene_minima, scene_maxima


def unproject_frustum_corners(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    depth: Float[Tensor, "#batch"],
) -> Float[Tensor, "batch 4 3"]:
    from src.geometry.projection import unproject

    device = extrinsics.device
    xy = torch.linspace(0, 1, 2, device=device)
    xy = torch.stack(torch.meshgrid(xy, xy, indexing="xy"), dim=-1)
    xy = rearrange(xy, "i j xy -> (i j) xy")[torch.tensor([0, 1, 3, 2], device=device)]

    directions = unproject(
        xy,
        torch.ones(1, dtype=torch.float32, device=device),
        rearrange(intrinsics, "b i j -> b () i j"),
    )
    directions = directions / directions[..., -1:]
    directions = einsum(extrinsics[..., :3, :3], directions, "b i j, b r j -> b r i")
    origins = rearrange(extrinsics[:, :3, 3], "b xyz -> b () xyz")
    depth = rearrange(depth, "b -> b () ()")
    return origins + depth * directions


