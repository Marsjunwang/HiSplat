from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from src.misc.image_io import save_image, unnormalize_images
from src.visualization.color_map import apply_color_map_to_image
from src.visualization.layout import add_border, hcat, vcat, resize


def _to_chw(image: Float[Tensor, "_ _ _"]) -> Float[Tensor, "3 h w"]:
    """Ensure image is 3xHxW float in [0,1]. Supports HxWx3, 3xHxW, or 1xHxW.
    Does not normalize or denormalize beyond channel replication.
    """
    if image.ndim == 3 and image.shape[0] in (1, 3):
        chw = image
    elif image.ndim == 3 and image.shape[-1] in (1, 3):
        chw = rearrange(image, "h w c -> c h w")
    else:
        raise AssertionError("image must be CHW or HWC with 1 or 3 channels")

    if chw.shape[0] == 1:
        chw = chw.repeat(3, 1, 1)
    return chw.clamp(0, 1).float()


def _maybe_denormalize(image: Tensor, assume_normalized: bool) -> Tensor:
    """Optionally convert from ImageNet normalization back to 0-1 RGB."""
    if assume_normalized:
        return unnormalize_images(image).clamp(0, 1)
    return image.clamp(0, 1)


def _select_channels(
    fmap: Float[Tensor, "c h w"],
    channels: Optional[Sequence[int]] = None,
    topk: Optional[int] = None,
) -> list[int]:
    c = fmap.shape[0]
    if channels is not None and len(channels) > 0:
        return [ch for ch in channels if 0 <= ch < c]
    if topk is not None and topk > 0:
        # Score channels by mean absolute activation
        scores = fmap.abs().mean(dim=(1, 2))
        k = min(topk, c)
        idx = torch.topk(scores, k=k, largest=True).indices
        return idx.tolist()
    # Default: first min(8,c) channels
    return list(range(min(8, c)))


def _normalize_per_channel(x: Tensor, eps: float = 1e-6) -> Tensor:
    # Normalize each channel to [0,1] by min-max
    c, h, w = x.shape
    x_ = x.view(c, -1)
    minv = x_.min(dim=1).values[:, None]
    maxv = x_.max(dim=1).values[:, None]
    rng = (maxv - minv).clamp_min(eps)
    x_ = (x_ - minv) / rng
    return x_.view(c, h, w)


def render_feature_channels(
    image: Float[Tensor, "3 h w"],
    feature: Float[Tensor, "c fh fw"],
    channels: Optional[Sequence[int]] = None,
    topk: Optional[int] = None,
    overlay: bool = False,
    grid_cols: int = 8,
    image_width: Optional[int] = 384,
    border: int = 4,
    title: Optional[str] = None,
    color_map: str = "inferno",
    image_is_normalized: bool = True,
) -> Float[Tensor, "3 H W"]:
    """Compose a visualization of the original image and chosen feature channels.

    - image: input image tensor, 3xHxW, range either [0,1] or ImageNet-normalized
    - feature: feature map tensor, CxHf x Wf
    - channels: explicit channel indices to visualize
    - topk: if channels is None, select top-k channels by mean |activation|
    - overlay: if True, upsample each channel, colorize and alpha-blend onto image
    - grid_cols: number of columns when laying out the grid of channels
    - image_width: resize final composite to this width (keeping aspect), None to keep
    - border: outer border size for panels
    - title: optional title text (rendered as label above)
    - color_map: matplotlib colormap name
    - image_is_normalized: True if `image` is ImageNet-normalized
    """
    # Prepare base image (denormalize if needed)
    base = _to_chw(image)
    base = _maybe_denormalize(base, assume_normalized=image_is_normalized)

    # Select and normalize feature channels
    selected = _select_channels(feature, channels, topk)
    fmap = feature[selected]
    fmap = _normalize_per_channel(fmap)

    # Upsample each channel to image size
    _, h, w = base.shape
    fmap_up = F.interpolate(fmap[None], size=(h, w), mode="bilinear", align_corners=False)[0]

    # Colorize feature channels
    colored = apply_color_map_to_image(fmap_up, color_map)
    # colored: [C_sel, 3, h, w]

    # Compose channel grid
    tiles: list[Tensor] = []
    for i in range(colored.shape[0]):
        tile = colored[i]
        if overlay:
            # Simple alpha blend with base; per-channel adaptive alpha via mean
            alpha = fmap_up[i].mean().clamp(0.2, 0.95)
            tile = (1 - alpha) * base + alpha * tile
        tiles.append(add_border(tile, border=border, color=0))

    if len(tiles) == 0:
        # fallback to base only
        grid = add_border(base, border=border, color=0)
    else:
        # Arrange grid by rows
        rows: list[Tensor] = []
        for start in range(0, len(tiles), grid_cols):
            row = hcat(*tiles[start : start + grid_cols], align="center", gap=4, gap_color=0)
            rows.append(row)
        grid = vcat(*rows, align="start", gap=4, gap_color=0)

    # Compose final side-by-side with original image
    left = add_border(base, border=border, color=0)
    right = grid
    composite = hcat(left, right, align="start", gap=8, gap_color=0)

    # Optional resize for convenience
    if image_width is not None:
        composite = resize(composite, width=image_width)

    return composite.clamp(0, 1)


def visualize_from_feature_list(
    image: Float[Tensor, "3 h w"],
    features: Sequence[Float[Tensor, "_ _ _"]],
    stage: int,
    channels: Optional[Sequence[int]] = None,
    topk: Optional[int] = None,
    overlay: bool = False,
    grid_cols: int = 8,
    image_width: Optional[int] = 384,
    border: int = 4,
    color_map: str = "inferno",
    image_is_normalized: bool = True,
) -> Float[Tensor, "3 H W"]:
    """Convenience wrapper: pick a stage from a list of features and render it."""
    assert 0 <= stage < len(features), f"stage {stage} out of range for {len(features)} features"
    return render_feature_channels(
        image=image,
        feature=features[stage],
        channels=channels,
        topk=topk,
        overlay=overlay,
        grid_cols=grid_cols,
        image_width=image_width,
        border=border,
        color_map=color_map,
        image_is_normalized=image_is_normalized,
    )


def save_feature_viz(
    image: Float[Tensor, "3 h w"],
    feature: Float[Tensor, "c _ _"],
    path: str | Path,
    channels: Optional[Sequence[int]] = None,
    topk: Optional[int] = None,
    overlay: bool = False,
    grid_cols: int = 8,
    image_width: Optional[int] = 768,
    border: int = 4,
    color_map: str = "inferno",
    image_is_normalized: bool = False,
) -> Path:
    """Render and save feature visualization to disk. Returns the output path."""
    img = render_feature_channels(
        image=image,
        feature=feature,
        channels=channels,
        topk=topk,
        overlay=overlay,
        grid_cols=grid_cols,
        image_width=image_width,
        border=border,
        color_map=color_map,
        image_is_normalized=image_is_normalized,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(img, path)
    return path


def save_feature_viz_from_list(
    image: Float[Tensor, "3 h w"],
    features: Sequence[Float[Tensor, "_ _ _"]],
    stage: int,
    path: str | Path,
    channels: Optional[Sequence[int]] = None,
    topk: Optional[int] = None,
    overlay: bool = False,
    grid_cols: int = 8,
    image_width: Optional[int] = 768,
    border: int = 4,
    color_map: str = "inferno",
    image_is_normalized: bool = False,
) -> Path:
    """Render and save feature visualization for a selected stage from list."""
    img = visualize_from_feature_list(
        image=image,
        features=features,
        stage=stage,
        channels=channels,
        topk=topk,
        overlay=overlay,
        grid_cols=grid_cols,
        image_width=image_width,
        border=border,
        color_map=color_map,
        image_is_normalized=image_is_normalized,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(img, path)
    return path


# from src.model.enhancer.utils.feature_viz import save_feature_viz, save_feature_viz_from_list

# # image: 3xH x W (ImageNet-normalized if coming from your pipeline)
# # features: list of feature tensors from encoder forward (each CxHf x Wf)

# # Example 1: visualize channels [0, 5, 7] of a single feature map
# save_feature_viz(
#     image=input_image[0],                 # 3xH x W
#     feature=features[2],                  # CxHf x Wf
#     path="/tmp/feat_stage2_ch_0_5_7.png",
#     channels=[0, 5, 7],
#     overlay=False,                        # set True to blend onto the image
#     color_map="inferno",                  # matplotlib cmap name
#     image_is_normalized=True              # set False if image is already 0-1 RGB
# )

# # Example 2: pick stage 3 and visualize top-8 channels by activation
# save_feature_viz_from_list(
#     image=input_image[0],
#     features=features,                    # list returned by encoder
#     stage=3,
#     path="/tmp/feat_stage3_top8.png",
#     topk=8,
#     overlay=True,
#     grid_cols=8,
#     image_width=768
# )