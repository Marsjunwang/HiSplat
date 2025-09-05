import os

import torch
import torch.nn.functional as F
from einops import rearrange

from .multiview_transformer import MultiViewFeatureTransformer
from .mvsformer_module.cost_volume import *
from .mvsformer_module.dino.dinov2 import vit_base
from .unimatch.backbone import ResUnet
from .unimatch.position import PositionEmbeddingSine
from .unimatch.utils import merge_splits, split_feature

# TODO: temporary config, should use cfg file
# The global vars
Align_Corners_Range = False

# for align feature
args = {
    "model_type": "DINOv2-base",
    "freeze_vit": True,
    "rescale": 1.0,
    "vit_ch": 768,
    "out_ch": 64,
    "vit_path": "./checkpoints/dinov2_vitb14_pretrain.pth",
    "pretrain_mvspp_path": "",
    "depth_type": ["ce", "ce", "ce", "ce"],
    "fusion_type": "cnn",
    "inverse_depth": True,
    "base_ch": [8, 8, 8, 8],
    "ndepths": [128, 64, 32, 16],
    "feat_chs": [32, 64, 128, 256],
    "depth_interals_ratio": [4.0, 2.67, 1.5, 1.0],
    "decoder_type": "CrossVITDecoder",
    "dino_cfg": {
        "use_flash2_dino": False,
        "softmax_scale": None,
        "train_avg_length": 762,
        "cross_interval_layers": 3,
        "decoder_cfg": {
            "init_values": 1.0,
            "prev_values": 0.5,
            "d_model": 768,
            "nhead": 12,
            "attention_type": "Linear",
            "ffn_type": "ffn",
            "softmax_scale": "entropy_invariance",
            "train_avg_length": 762,
            "self_cross_types": None,
            "post_norm": False,
            "pre_norm_query": True,
            "no_combine_norm": False,
        },
    },
    "FMT_config": {
        "attention_type": "Linear",
        "base_channel": 8 * 4,
        "d_model": 64 * 4,
        "nhead": 4,
        "init_values": 1.0,
        "layer_names": ["self", "cross", "self", "cross"],
        "ffn_type": "ffn",
        "softmax_scale": "entropy_invariance",
        "train_avg_length": 12185,
        "attn_backend": "FLASH2",
        "self_cross_types": None,
        "post_norm": False,
        "pre_norm_query": False,
    },
    "cost_reg_type": ["PureTransformerCostReg", "Normal", "Normal", "Normal"],
    "use_pe3d": True,
    "transformer_config": [
        {
            "base_channel": 8 * 4,
            "mid_channel": 64 * 4,
            "num_heads": 4,
            "down_rate": [2, 4, 4],
            "mlp_ratio": 4.0,
            "layer_num": 6,
            "drop": 0.0,
            "attn_drop": 0.0,
            "position_encoding": True,
            "attention_type": "FLASH2",
            "softmax_scale": "entropy_invariance",
            "train_avg_length": 12185,
            "use_pe_proj": True,
        }
    ],
}


def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        features_splits = [split_feature(x, num_splits=attn_splits) for x in features_list]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [merge_splits(x, num_splits=attn_splits) for x in features_splits]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return out_features_list


class BackbonePyramid(torch.nn.Module):
    """docstring for BackboneMultiview.
    This function is used to extract the feature of different view
    the CNN is used to extract single view feature
    Transformer is used to extract single&multi view feature
    
    Visualization Usage:
    - Set model.visualize_dino = True to visualize DINO features
    - Set model.visualize_backbone = True to visualize backbone output features  
    - Set model.visualize_trans = True to visualize transformer output features
    - Visualizations will be saved to 'debug_visualizations/' directory
    """

    def __init__(
        self,
        feature_channels=128,
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        no_self_attn=False,
        no_cross_attn=False,
        num_head=1,
        no_split_still_shift=False,
        no_ffn=False,
        global_attn_fast=True,
        downscale_factor=8,
        use_epipolar_trans=False,
    ):
        super(BackbonePyramid, self).__init__()
        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            no_cross_attn=no_cross_attn,
        )
        self.backbone = ResUnet()
        self.dino = DinoExtractor()

    def normalize_images(self, images):
        # TODO: should use the normalize for other model
        """Normalize image to match the pretrained GMFlow backbone.
        images: (B, N_Views, C, H, W)
        """
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, context):
        # stage 1-> 32x32 stage 2-> 64x64 stage 3-> 128x128 stage 4-> 256x256
        # context = self.gen_all_rs_combination(context)
        imgs = self.normalize_images(context["image"])
        org_imgs = imgs
        B, V, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[3], imgs.shape[4]
        # dinov2 patchsize=14,  0.5 * 14/16
        imgs = imgs.reshape(B * V, 3, H, W)
        dino_feature = self.dino(org_imgs)
        
        # save context image for comapre
        if hasattr(self, 'visualize_context') and self.visualize_context:
            self._visualize_context(context)
            
        # Visualize dino features for debugging
        if hasattr(self, 'visualize_dino') and self.visualize_dino:
            self._visualize_dino_features(dino_feature)
        
        out_feature = self.backbone(imgs, dino_feature)
        
        # Visualize out_feature for debugging
        if hasattr(self, 'visualize_backbone') and self.visualize_backbone:
            self._visualize_backbone_features(out_feature, B, V, viz_view=0)
        
        trans_feature_in = rearrange(out_feature[0], "(b v) c h w -> b v c h w", b=B).chunk(dim=1, chunks=V)
        trans_feature_in = [f[:, 0] for f in trans_feature_in]
        # add position to features
        trans_feature_in = feature_add_position_list(trans_feature_in, 2, out_feature[0].size(1))
        cur_features_list = self.transformer(trans_feature_in, 2)
        trans_features = torch.stack(cur_features_list, dim=1)  # [1, 2, 128, 64, 64]
        
        # Visualize trans_features for debugging
        if hasattr(self, 'visualize_trans') and self.visualize_trans:
            self._visualize_trans_features(trans_features, viz_view=0)
        
        return (out_feature, trans_features)

    def _visualize_context(self, context):
        """Save input context images for quick visual debugging.
        Expects context['image'] of shape [B, V, C, H, W] with values in [0, 1]."""
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        os.makedirs('debug_visualizations', exist_ok=True)
        if "image" not in context:
            return
        images = context["image"].detach().cpu()
        if images.ndim != 5:
            print(f"Unexpected context['image'] ndim={images.ndim}, expected 5")
            return
        B, V, C, H, W = images.shape
        print(f"Context image shape: {tuple(images.shape)}  range: [{images.min().item():.4f}, {images.max().item():.4f}]")
        # Clamp to [0,1] for visualization
        images = images.clamp(0, 1)
        max_batches_to_save = min(B, 2)
        for b in range(max_batches_to_save):
            fig, axes = plt.subplots(1, V, figsize=(5 * V, 5))
            if isinstance(axes, np.ndarray):
                axes_list = list(axes.flatten())
            else:
                axes_list = [axes]
            for v in range(V):
                img = images[b, v].permute(1, 2, 0).numpy()  # H, W, C
                ax = axes_list[v]
                ax.imshow(img)
                ax.set_title(f'b{b} v{v}')
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'debug_visualizations/context_b{b}.png', dpi=150, bbox_inches='tight')
            plt.close()

    def _visualize_dino_features(self, dino_feature):
        """Visualize DINO features for debugging"""
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        os.makedirs('debug_visualizations', exist_ok=True)
        
        # Visualize first batch, first view's dino features
        feat_vis = dino_feature[0].detach().cpu()  # [64, 32, 32]
        
        # Check if we have spatial dimensions for visualization
        if len(feat_vis.shape) >= 2:
            # Show first few channels
            fig, axes = plt.subplots(4, 8, figsize=(16, 8))
            for i in range(min(32, feat_vis.shape[0])):
                ax = axes[i // 8, i % 8]
                if len(feat_vis.shape) == 3:  # [C, H, W]
                    im = ax.imshow(feat_vis[i], cmap='viridis')
                elif len(feat_vis.shape) == 2:  # [H, W] 
                    im = ax.imshow(feat_vis, cmap='viridis')
                    break  # Only one image to show
                else:  # [C] - 1D features, create a bar plot instead
                    ax.bar(range(len(feat_vis)), feat_vis)
                    ax.set_title(f'Feature values')
                    break
                ax.set_title(f'Channel {i}')
                ax.axis('off')
                plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            plt.savefig('debug_visualizations/dino_features_channels.png', dpi=150)
            plt.close()
        
        # Visualize feature statistics
        print(f"DINO feature shape: {dino_feature.shape}")
        print(f"DINO feature range: [{dino_feature.min().item():.4f}, {dino_feature.max().item():.4f}]")
        print(f"DINO feature mean: {dino_feature.mean().item():.4f}")
        print(f"DINO feature std: {dino_feature.std().item():.4f}")
        
        # Save feature norm visualization with hybrid top selection overlays
        feat_norm = torch.norm(feat_vis, dim=0)  # [H, W]
        top_n = 256
        border_ratio = 0.05

        # 2x2 panel similar to other visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Regular norm visualization
        im1 = ax1.imshow(feat_norm, cmap='hot')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('DINO Features - L2 Norm')

        # 2. Hybrid top selection: part global, part per-region
        topn_indices_flat, topn_mask, topn_coords = self._select_top_indices_hybrid(
            feat_norm,
            top_n=top_n,
            border_ratio=border_ratio,
            global_ratio=0.125,
            region_rows=8,
            region_cols=8,
        )

        # Use a discrete colormap for better distinction
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, min(top_n, 20)))

        # Create custom visualization showing top N with different colors
        topn_visual = np.zeros((*feat_norm.shape, 3))  # RGB image
        for i, coord in enumerate(topn_coords):
            y, x = coord[0].item(), coord[1].item()
            color_idx = i % len(colors)
            topn_visual[y, x] = colors[color_idx][:3]  # Use RGB, ignore alpha

        ax2.imshow(topn_visual)
        ax2.set_title(f'Top {top_n} Norm Values (Excluding 5% Border, Different Colors)')
        ax2.axis('off')

        # 3. Norm values with top N highlighted in overlay
        im3 = ax3.imshow(feat_norm, cmap='gray', alpha=0.7)

        # Draw border exclusion area
        H, W = feat_norm.shape
        border_h = max(1, int(H * border_ratio))
        border_w = max(1, int(W * border_ratio))
        ax3.add_patch(plt.Rectangle((0, 0), W, border_h, fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7))
        ax3.add_patch(plt.Rectangle((0, H-border_h), W, border_h, fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7))
        ax3.add_patch(plt.Rectangle((0, 0), border_w, H, fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7))
        ax3.add_patch(plt.Rectangle((W-border_w, 0), border_w, H, fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7))

        # Overlay top N positions with colored dots
        for i, coord in enumerate(topn_coords):
            y, x = coord[0].item(), coord[1].item()
            color_idx = i % len(colors)
            ax3.scatter(x, y, c=[colors[color_idx]], s=50, marker='o', edgecolors='white', linewidth=1)
            if i < 20:  # Only show numbers for first 20
                ax3.text(x, y, str(i+1), fontsize=8, ha='center', va='center', color='white', weight='bold')

        plt.colorbar(im3, ax=ax3)
        ax3.set_title(f'Norm with Top {top_n} Highlighted (Excluding 5% Border, Numbers 1-20)')

        # 4. Bar chart of top N norm values (show only first 20 for readability)
        if len(topn_indices_flat) > 0:
            feat_norm_flat = feat_norm.flatten()
            topn_values = feat_norm_flat[topn_indices_flat].detach().cpu().numpy()
            display_count = min(20, len(topn_values))
            bars = ax4.bar(range(display_count), topn_values[:display_count], 
                          color=[colors[i % len(colors)] for i in range(display_count)])
            ax4.set_title(f'Top {display_count}/{top_n} Norm Values (Excluding 5% Border)')
            ax4.set_xlabel('Rank')
            ax4.set_ylabel('Norm Value')
            ax4.grid(True, alpha=0.3)
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, topn_values[:display_count])):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No valid values found', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title(f'Top {top_n} Norm Values (Excluding 5% Border) - No Data')

        plt.tight_layout()
        plt.savefig('debug_visualizations/dino_feature_norm.png', dpi=150)
        plt.close()

        # Print top N norm values and their positions (show first 20)
        print(f"\nDINO - Top {top_n} norm values (excluding 5% border, showing first 20):")
        if len(topn_indices_flat) > 0:
            feat_norm_flat = feat_norm.flatten()
            for i, (idx, coord) in enumerate(zip(topn_indices_flat[:20], topn_coords[:20])):
                y, x = coord[0].item(), coord[1].item()
                val = feat_norm_flat[idx].item()
                print(f"  Rank {i+1}: Value={val:.4f}, Position=({y}, {x})")
        else:
            print("  No valid values found after excluding border pixels")

    def _select_top_indices_hybrid(
        self,
        feat_norm: torch.Tensor,
        top_n: int,
        border_ratio: float = 0.05,
        global_ratio: float = 0.5,
        region_rows: int = 8,
        region_cols: int = 8,
    ):
        """Select top indices by combining global top-k and per-region top-k.

        - Excludes a border defined by border_ratio on each side
        - Picks round(top_n * global_ratio) globally
        - Splits the rest evenly across a grid of region_rows x region_cols
        - Ensures unique indices across selections and handles insufficient valid points
        Returns (topn_indices_flat, topn_mask, topn_coords)
        """
        H, W = feat_norm.shape
        device = feat_norm.device

        # Build border mask
        border_h = max(1, int(H * border_ratio))
        border_w = max(1, int(W * border_ratio))
        valid_mask = torch.ones((H, W), dtype=torch.bool, device=device)
        valid_mask[:border_h, :] = False
        valid_mask[-border_h:, :] = False
        valid_mask[:, :border_w] = False
        valid_mask[:, -border_w:] = False

        feat_norm_flat = feat_norm.flatten()
        valid_mask_flat = valid_mask.flatten()

        # Determine budgets
        top_n = int(max(0, top_n))
        if top_n == 0:
            empty = torch.tensor([], dtype=torch.long, device=device)
            empty_mask = torch.zeros_like(valid_mask_flat, dtype=torch.bool)
            return empty, empty_mask.reshape(H, W), torch.zeros((0, 2), dtype=torch.long)

        global_budget = int(round(top_n * global_ratio))
        region_budget = max(0, top_n - global_budget)

        selected_mask_flat = torch.zeros_like(valid_mask_flat, dtype=torch.bool)

        # Global selection
        if global_budget > 0:
            # Mask invalid positions by setting to -inf
            scores = feat_norm_flat.clone()
            scores[~valid_mask_flat] = -float('inf')
            k = min(global_budget, int(valid_mask_flat.sum().item()))
            if k > 0:
                global_indices = torch.topk(scores, k=k).indices
                selected_mask_flat[global_indices] = True

        # Per-region selection
        if region_budget > 0:
            # Adjust grid if too fine for current H, W
            region_rows_eff = max(1, min(region_rows, H))
            region_cols_eff = max(1, min(region_cols, W))
            # Roughly even allocation
            per_region = region_budget // (region_rows_eff * region_cols_eff)
            remainder = region_budget - per_region * region_rows_eff * region_cols_eff

            # Build exact integer partitions to avoid rounding bias from linspace
            base_h = H // region_rows_eff
            extra_h = H % region_rows_eff
            row_heights = [base_h + 1 if i < extra_h else base_h for i in range(region_rows_eff)]
            y_starts = [0]
            for rh in row_heights:
                y_starts.append(y_starts[-1] + rh)

            base_w = W // region_cols_eff
            extra_w = W % region_cols_eff
            col_widths = [base_w + 1 if i < extra_w else base_w for i in range(region_cols_eff)]
            x_starts = [0]
            for cw in col_widths:
                x_starts.append(x_starts[-1] + cw)

            idx_list = []
            for ry in range(region_rows_eff):
                for rx in range(region_cols_eff):
                    y0, y1 = y_starts[ry], y_starts[ry + 1]
                    x0, x1 = x_starts[rx], x_starts[rx + 1]
                    region_h = max(1, y1 - y0)
                    region_w = max(1, x1 - x0)
                    # Build region mask
                    region_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
                    region_mask[y0:y0 + region_h, x0:x0 + region_w] = True
                    # Valid and not already selected
                    mask = region_mask & valid_mask & (~selected_mask_flat.reshape(H, W))
                    mask_flat = mask.flatten()
                    # Quota for this region (distribute remainders first regions)
                    k_region = per_region + (1 if remainder > 0 else 0)
                    if remainder > 0:
                        remainder -= 1
                    if k_region <= 0:
                        continue
                    # Get top-k within region
                    scores = feat_norm_flat.clone()
                    scores[~mask_flat] = -float('inf')
                    k_take = min(k_region, int(mask_flat.sum().item()))
                    if k_take > 0:
                        region_top = torch.topk(scores, k=k_take).indices
                        idx_list.append(region_top)
                        selected_mask_flat[region_top] = True

            if len(idx_list) > 0:
                region_indices = torch.cat(idx_list, dim=0)
            else:
                region_indices = torch.tensor([], dtype=torch.long, device=device)
        else:
            region_indices = torch.tensor([], dtype=torch.long, device=device)

        # Combine
        topn_indices_flat = torch.nonzero(selected_mask_flat, as_tuple=False).flatten()
        # If still short due to insufficient valid points, backfill by next best overall
        if topn_indices_flat.numel() < top_n:
            deficit = top_n - topn_indices_flat.numel()
            scores = feat_norm_flat.clone()
            scores[~valid_mask_flat] = -float('inf')
            scores[selected_mask_flat] = -float('inf')
            k_backfill = min(deficit, int((~torch.isinf(scores)).sum().item()))
            if k_backfill > 0:
                backfill = torch.topk(scores, k=k_backfill).indices
                selected_mask_flat[backfill] = True
                topn_indices_flat = torch.nonzero(selected_mask_flat, as_tuple=False).flatten()

        # Build mask and coords
        topn_mask = torch.zeros_like(valid_mask_flat, dtype=torch.bool)
        topn_mask[topn_indices_flat] = True
        topn_mask_2d = topn_mask.reshape(H, W)
        y_coords, x_coords = torch.nonzero(topn_mask_2d, as_tuple=True)
        topn_coords = torch.stack([y_coords, x_coords], dim=1)
        return topn_indices_flat, topn_mask_2d, topn_coords

    def _visualize_backbone_features(self, out_feature, batch_size, num_views, viz_view=0):
        """Visualize backbone output features for debugging"""
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        os.makedirs('debug_visualizations', exist_ok=True)
        
        # out_feature is a list/tuple of features at different scales
        for scale_idx, scale_feat in enumerate(out_feature):
            if scale_feat is None:
                continue
                
            # Reshape from (B*V, C, H, W) to (B, V, C, H, W) for visualization
            _, C, H, W = scale_feat.shape
            scale_feat_reshaped = scale_feat.view(batch_size, num_views, C, H, W)
            
            # Visualize first batch, first view
            feat_vis = scale_feat_reshaped[0, viz_view].detach().cpu()  # [C, H, W]
            
            print(f"Backbone feature scale {scale_idx} shape: {scale_feat.shape}")
            print(f"Backbone feature scale {scale_idx} range: [{scale_feat.min().item():.4f}, {scale_feat.max().item():.4f}]")
            print(f"Backbone feature scale {scale_idx} mean: {scale_feat.mean().item():.4f}")
            print(f"Backbone feature scale {scale_idx} std: {scale_feat.std().item():.4f}")
            
            # Visualize feature channels (show first 16 channels in a 4x4 grid)
            num_channels_to_show = min(16, feat_vis.shape[0])
            if num_channels_to_show > 0:
                rows = 4
                cols = 4
                fig, axes_raw = plt.subplots(rows, cols, figsize=(12, 12))
                
                # Convert to list for consistent handling
                if rows * cols == 1:
                    axes_list = [axes_raw]
                else:
                    axes_list = list(axes_raw.flatten())
                
                for i in range(num_channels_to_show):
                    ax = axes_list[i]
                    im = ax.imshow(feat_vis[i], cmap='viridis')
                    ax.set_title(f'Ch {i}')
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Hide unused subplots
                for i in range(num_channels_to_show, len(axes_list)):
                    axes_list[i].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'debug_visualizations/backbone_features_scale{scale_idx}_channels.png', dpi=150, bbox_inches='tight')
                plt.close()
            
            # Visualize feature norm across channels with top N values highlighted
            # Use formula: 20 * 2^scale_idx
            top_n = 256
            
            feat_norm = torch.norm(feat_vis, dim=0)  # [H, W]
            
            # Create border mask to exclude edge pixels (5% border on each side)
            border_ratio = 0.05
            border_h = max(1, int(H * border_ratio))
            border_w = max(1, int(W * border_ratio))
            
            # Create mask excluding border pixels
            border_mask = torch.ones_like(feat_norm, dtype=torch.bool)
            border_mask[:border_h, :] = False  # top border
            border_mask[-border_h:, :] = False  # bottom border
            border_mask[:, :border_w] = False  # left border
            border_mask[:, -border_w:] = False  # right border
            
            # Apply border mask to norm values for analysis
            feat_norm_masked = feat_norm.clone()
            feat_norm_masked[~border_mask] = -float('inf')  # Set border pixels to very low value so they won't be selected
            
            # Create a figure with multiple visualizations
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Regular norm visualization
            im1 = ax1.imshow(feat_norm, cmap='hot')
            plt.colorbar(im1, ax=ax1)
            ax1.set_title(f'Backbone Features Scale {scale_idx} - L2 Norm')
            
            # 2. Hybrid top selection: part global, part per-region
            topn_indices_flat, topn_mask, topn_coords = self._select_top_indices_hybrid(
                feat_norm,
                top_n=top_n,
                border_ratio=border_ratio,
                global_ratio=0.125,
                region_rows=8,
                region_cols=8,
            )
            
            # Create color-coded visualization for top N values
            norm_colored = feat_norm.clone()
            norm_colored[~topn_mask] = 0  # Set non-topN to 0
            
            # Use a discrete colormap for better distinction
            colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, min(top_n, 20)))  # Use tab20 colormap for distinct colors
            
            # Create custom visualization showing top N with different colors
            topn_visual = np.zeros((*feat_norm.shape, 3))  # RGB image
            for i, coord in enumerate(topn_coords):
                y, x = coord[0].item(), coord[1].item()
                color_idx = i % len(colors)
                topn_visual[y, x] = colors[color_idx][:3]  # Use RGB, ignore alpha
            
            ax2.imshow(topn_visual)
            ax2.set_title(f'Top {top_n} Norm Values (Excluding 5% Border, Different Colors)')
            ax2.axis('off')
            
            # 3. Norm values with top N highlighted in overlay
            im3 = ax3.imshow(feat_norm, cmap='gray', alpha=0.7)
            
            # Draw border exclusion area
            ax3.add_patch(plt.Rectangle((0, 0), W, border_h, fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7))
            ax3.add_patch(plt.Rectangle((0, H-border_h), W, border_h, fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7))
            ax3.add_patch(plt.Rectangle((0, 0), border_w, H, fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7))
            ax3.add_patch(plt.Rectangle((W-border_w, 0), border_w, H, fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7))
            
            # Overlay top N positions with colored dots
            for i, coord in enumerate(topn_coords):
                y, x = coord[0].item(), coord[1].item()
                color_idx = i % len(colors)
                ax3.scatter(x, y, c=[colors[color_idx]], s=50, marker='o', edgecolors='white', linewidth=1)
                if i < 20:  # Only show numbers for first 20
                    ax3.text(x, y, str(i+1), fontsize=8, ha='center', va='center', color='white', weight='bold')
            
            plt.colorbar(im3, ax=ax3)
            ax3.set_title(f'Norm with Top {top_n} Highlighted (Excluding 5% Border, Numbers 1-20)')
            
            # 4. Bar chart of top N norm values (show only first 20 for readability)
            if len(topn_indices_flat) > 0:
                feat_norm_flat = feat_norm.flatten()
                topn_values = feat_norm_flat[topn_indices_flat].detach().cpu().numpy()
                display_count = min(20, len(topn_values))
                bars = ax4.bar(range(display_count), topn_values[:display_count], 
                              color=[colors[i % len(colors)] for i in range(display_count)])
                ax4.set_title(f'Top {display_count}/{top_n} Norm Values (Excluding 5% Border)')
                ax4.set_xlabel('Rank')
                ax4.set_ylabel('Norm Value')
                ax4.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, topn_values[:display_count])):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                            f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
            else:
                ax4.text(0.5, 0.5, 'No valid values found', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title(f'Top {top_n} Norm Values (Excluding 5% Border) - No Data')
            
            plt.tight_layout()
            plt.savefig(f'debug_visualizations/backbone_features_scale{scale_idx}_norm_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Print top N norm values and their positions (show first 20)
            print(f"\nScale {scale_idx} - Top {top_n} norm values (excluding 5% border, showing first 20):")
            if len(topn_indices_flat) > 0:
                feat_norm_flat = feat_norm.flatten()
                for i, (idx, coord) in enumerate(zip(topn_indices_flat[:20], topn_coords[:20])):
                    y, x = coord[0].item(), coord[1].item()
                    val = feat_norm_flat[idx].item()
                    print(f"  Rank {i+1}: Value={val:.4f}, Position=({y}, {x})")
            else:
                print("  No valid values found after excluding border pixels")
            
            # Visualize feature mean across channels
            feat_mean = torch.mean(feat_vis, dim=0)  # [H, W]
            plt.figure(figsize=(8, 6))
            im = plt.imshow(feat_mean, cmap='RdBu_r')
            plt.colorbar(im)
            plt.title(f'Backbone Features Scale {scale_idx} - Channel Mean')
            plt.savefig(f'debug_visualizations/backbone_features_scale{scale_idx}_mean.png', dpi=150, bbox_inches='tight')
            plt.close()

    def _visualize_trans_features(self, trans_features, viz_view=0):
        """Visualize transformer output features for debugging"""
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        os.makedirs('debug_visualizations', exist_ok=True)
        
        # trans_features is [B, V, C, H, W]
        B, V, C, H, W = trans_features.shape
        trans_features_reshaped = trans_features.view(B, V, C, H, W)
        
        # Visualize first batch, first view
        feat_vis = trans_features_reshaped[0, viz_view].detach().cpu() # [C, H, W]
        
        print(f"Transformer feature shape: {trans_features.shape}")
        print(f"Transformer feature range: [{trans_features.min().item():.4f}, {trans_features.max().item():.4f}]")
        print(f"Transformer feature mean: {trans_features.mean().item():.4f}")
        print(f"Transformer feature std: {trans_features.std().item():.4f}")
        
        # Visualize feature channels (show first 16 channels in a 4x4 grid)
        num_channels_to_show = min(16, feat_vis.shape[0])
        if num_channels_to_show > 0:
            rows = 4
            cols = 4
            fig, axes_raw = plt.subplots(rows, cols, figsize=(12, 12))
            
            # Convert to list for consistent handling
            if isinstance(axes_raw, np.ndarray):
                axes_list = list(axes_raw.flatten())
            else:
                axes_list = [axes_raw]
            
            for i in range(num_channels_to_show):
                ax = axes_list[i]
                im = ax.imshow(feat_vis[i], cmap='viridis')
                ax.set_title(f'Ch {i}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Hide unused subplots
            for i in range(num_channels_to_show, len(axes_list)):
                axes_list[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'debug_visualizations/transformer_features_channels.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # Visualize feature norm across channels with top N values highlighted (hybrid selection)
        top_n = 256
        feat_norm = torch.norm(feat_vis, dim=0) # [H, W]
        border_ratio = 0.05
        # Create a figure with multiple visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Regular norm visualization
        im1 = ax1.imshow(feat_norm, cmap='hot')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title(f'Transformer Features - L2 Norm')
        
        # 2. Hybrid top selection: part global, part per-region
        topn_indices_flat, topn_mask, topn_coords = self._select_top_indices_hybrid(
            feat_norm,
            top_n=top_n,
            border_ratio=border_ratio,
            global_ratio=0.125,
            region_rows=8,
            region_cols=8,
        )
        
        # Create color-coded visualization for top N values
        norm_colored = feat_norm.clone()
        norm_colored[~topn_mask] = 0 # Set non-topN to 0
        
        # Use a discrete colormap for better distinction
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, min(top_n, 20))) # Use tab20 colormap for distinct colors
        
        # Create custom visualization showing top N with different colors
        topn_visual = np.zeros((*feat_norm.shape, 3)) # RGB image
        
        for i, coord in enumerate(topn_coords):
            y, x = coord[0].item(), coord[1].item()
            color_idx = i % len(colors)
            topn_visual[y, x] = colors[color_idx][:3] # Use RGB, ignore alpha
        
        ax2.imshow(topn_visual)
        ax2.set_title(f'Top {top_n} Norm Values (Excluding 5% Border, Different Colors)')
        ax2.axis('off')
        
        # 3. Norm values with top N highlighted in overlay
        im3 = ax3.imshow(feat_norm, cmap='gray', alpha=0.7)
        
        # Draw border exclusion area
        H, W = feat_norm.shape
        border_h = max(1, int(H * border_ratio))
        border_w = max(1, int(W * border_ratio))
        ax3.add_patch(plt.Rectangle((0, 0), W, border_h, fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7))
        ax3.add_patch(plt.Rectangle((0, H-border_h), W, border_h, fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7))
        ax3.add_patch(plt.Rectangle((0, 0), border_w, H, fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7))
        ax3.add_patch(plt.Rectangle((W-border_w, 0), border_w, H, fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7))
        
        # Overlay top N positions with colored dots
        for i, coord in enumerate(topn_coords):
            y, x = coord[0].item(), coord[1].item()
            color_idx = i % len(colors)
            ax3.scatter(x, y, c=[colors[color_idx]], s=50, marker='o', edgecolors='white', linewidth=1)
            if i < 20: # Only show numbers for first 20
                ax3.text(x, y, str(i+1), fontsize=8, ha='center', va='center', color='white', weight='bold')
        
        plt.colorbar(im3, ax=ax3)
        ax3.set_title(f'Norm with Top {top_n} Highlighted (Excluding 5% Border, Numbers 1-20)')
        
        # 4. Bar chart of top N norm values (show only first 20 for readability)
        if len(topn_indices_flat) > 0:
            feat_norm_flat = feat_norm.flatten()
            topn_values = feat_norm_flat[topn_indices_flat].detach().cpu().numpy()
            display_count = min(20, len(topn_values))
            bars = ax4.bar(range(display_count), topn_values[:display_count], 
                          color=[colors[i % len(colors)] for i in range(display_count)])
            ax4.set_title(f'Top {display_count}/{top_n} Norm Values (Excluding 5% Border)')
            ax4.set_xlabel('Rank')
            ax4.set_ylabel('Norm Value')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, topn_values[:display_count])):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No valid values found', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title(f'Top {top_n} Norm Values (Excluding 5% Border) - No Data')
        
        plt.tight_layout()
        plt.savefig(f'debug_visualizations/transformer_features_norm_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print top N norm values and their positions (show first 20)
        print(f"\nTransformer - Top {top_n} norm values (excluding 5% border, showing first 20):")
        if len(topn_indices_flat) > 0:
            feat_norm_flat = feat_norm.flatten()
            for i, (idx, coord) in enumerate(zip(topn_indices_flat[:20], topn_coords[:20])):
                y, x = coord[0].item(), coord[1].item()
                val = feat_norm_flat[idx].item()
                print(f"  Rank {i+1}: Value={val:.4f}, Position=({y}, {x})")
        else:
            print("  No valid values found after excluding border pixels")
        
        # Visualize feature mean across channels
        feat_mean = torch.mean(feat_vis, dim=0) # [H, W]
        plt.figure(figsize=(8, 6))
        im = plt.imshow(feat_mean, cmap='RdBu_r')
        plt.colorbar(im)
        plt.title(f'Transformer - Channel Mean')
        plt.savefig(f'debug_visualizations/transformer_features_mean.png', dpi=150, bbox_inches='tight')
        plt.close()

    def forward(
        self,
        context,
        attn_splits=2,
        return_cnn_features=False,
        epipolar_kwargs=None,
    ):
        """images: (B, N_Views, C, H, W), range [0, 1]"""
        # resolution low to high
        features_list = self.extract_feature(context=context)  # list of features

        return features_list

    def gen_proj_matrices(self, context):
        # extri and intri
        # !!!Attention: Mvsformer++ need w2c so we have to inverse
        inverse = True
        extri = context["extrinsics"].clone().inverse() if inverse else context["extrinsics"].clone()
        intri = context["intrinsics"].clone()

        b, v, _, _ = intri.shape
        _, _, _, h, w = context["image"].shape
        intri = torch.cat(
            [
                torch.cat([intri, torch.zeros((b, v, 1, 3), device=intri.device)], dim=2),
                torch.zeros((b, v, 4, 1), device=intri.device),
            ],
            dim=3,
        )
        intri_org = intri.clone()
        proj_matrices = {}  # Initialize as empty dict instead of dict with int values
        # four stages and the intrinsics scale by 2x
        for i in range(1, 5):
            intri[..., 0, :] = intri_org[..., 0, :] * w / 2 ** (4 - i)
            intri[..., 1, :] = intri_org[..., 1, :] * h / 2 ** (4 - i)
            proj_matrices["stage{}".format(i)] = torch.stack([extri, intri], dim=2)
        return proj_matrices  # [b, v, 2, 4, 4]

    def gen_all_rs_combination(self, context):
        # according to the input context, output all the reference and source combination of different pictures
        def reverse_and_add_to_bv(input_tensor):
            # input_tensor: [b, v, ...]
            b, v, *_ = input_tensor.shape
            tensor_return_order_list = []
            for i in range(v):
                range_v = list(range(v))
                del range_v[i]
                tensor_return_order_list += [[i] + range_v]
            return_input_tensor = rearrange(
                input_tensor[:, tensor_return_order_list], "b com_n v ... -> (b com_n) v ..."
            )
            return return_input_tensor

        return {k: reverse_and_add_to_bv(v) for k, v in context.items()}


class DinoExtractor(torch.nn.Module):
    def __init__(self, vit_path="./checkpoints/dinov2_vitb14_pretrain.pth"):
        super(DinoExtractor, self).__init__()
        self.vit_cfg = {
            "model_type": "DINOv2-base",
            "freeze_vit": True,
            "rescale": 1.0,
            "vit_ch": 768,
            "out_ch": 64,
            "vit_path": "./checkpoints/dinov2_vitb14_pretrain.pth",
            "pretrain_mvspp_path": "",
            "depth_type": ["ce", "ce", "ce", "ce"],
            "fusion_type": "cnn",
            "inverse_depth": True,
            "base_ch": [8, 8, 8, 8],
            "ndepths": [128, 64, 32, 16],
            "feat_chs": [32, 64, 128, 256],
            "depth_interals_ratio": [4.0, 2.67, 1.5, 1.0],
            "decoder_type": "CrossVITDecoder",
            "dino_cfg": {
                "use_flash2_dino": False,
                "softmax_scale": None,
                "train_avg_length": 762,
                "cross_interval_layers": 3,
                "decoder_cfg": {
                    "init_values": 1.0,
                    "prev_values": 0.5,
                    "d_model": 768,
                    "nhead": 12,
                    "attention_type": "Linear",
                    "ffn_type": "ffn",
                    "softmax_scale": "entropy_invariance",
                    "train_avg_length": 762,
                    "self_cross_types": None,
                    "post_norm": False,
                    "pre_norm_query": True,
                    "no_combine_norm": False,
                },
            },
            "FMT_config": {
                "attention_type": "Linear",
                "base_channel": 8 * 4,
                "d_model": 64 * 4,
                "nhead": 4,
                "init_values": 1.0,
                "layer_names": ["self", "cross", "self", "cross"],
                "ffn_type": "ffn",
                "softmax_scale": "entropy_invariance",
                "train_avg_length": 12185,
                "attn_backend": "FLASH2",
                "self_cross_types": None,
                "post_norm": False,
                "pre_norm_query": False,
            },
            "cost_reg_type": ["PureTransformerCostReg", "Normal", "Normal", "Normal"],
            "use_pe3d": True,
            "transformer_config": [
                {
                    "base_channel": 8 * 4,
                    "mid_channel": 64 * 4,
                    "num_heads": 4,
                    "down_rate": [2, 4, 4],
                    "mlp_ratio": 4.0,
                    "layer_num": 6,
                    "drop": 0.0,
                    "attn_drop": 0.0,
                    "position_encoding": True,
                    "attention_type": "FLASH2",
                    "softmax_scale": "entropy_invariance",
                    "train_avg_length": 12185,
                    "use_pe_proj": True,
                }
            ],
        }
        dino_cfg = args.get("dino_cfg", {})
        self.vit = vit_base(img_size=518, patch_size=14, init_values=1.0, block_chunks=0, ffn_layer="mlp", **dino_cfg)
        self.decoder_vit = CrossVITDecoder(self.vit_cfg)
        if os.path.exists(vit_path):
            state_dict = torch.load(vit_path, map_location="cpu")
            from .mvsformer_module.utils import torch_init_model

            torch_init_model(self.vit, state_dict, key="model")
            print("!!!Successfully load the DINOV2 ckpt from", vit_path)
        else:
            print("!!!No weight in", vit_path, "testing should neglect this.")

    def forward(self, image):
        rescale = 0.4375
        b, v, c, h, w = image.shape
        image = rearrange(image, "b v c h w -> (b v) c h w")
        vit_h, vit_w = int(h * rescale // 14 * 14), int(w * rescale // 14 * 14)
        vit_imgs = F.interpolate(image, (vit_h, vit_w), mode="bicubic", align_corners=False)
        with torch.no_grad():
            vit_out = self.vit.forward_interval_features(vit_imgs)
        vit_out = [vi.reshape(b, v, -1, self.vit.embed_dim) for vi in vit_out]
        vit_shape = [b, v, vit_h // self.vit.patch_size, vit_w // self.vit.patch_size, self.vit.embed_dim]
        vit_feat = self.decoder_vit.forward(vit_out, Fmats=None, vit_shape=vit_shape)
        return vit_feat
