import os
import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import Trainer

from jaxtyping import install_import_hook

import torch.nn.functional as F

# Ensure typed modules are importable with beartype/jaxtyping
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg, get_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper

import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging


def acos_clamped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.acos(x.clamp(min=-1.0 + eps, max=1.0 - eps))


def pose_metrics_from_se3(pred_01: torch.Tensor, gt_01: torch.Tensor):
    """
    Compute geodesic rotation error (deg) and translation L2 from two SE(3) transforms.
    pred_01, gt_01: [B,4,4]
    Returns scalars (rot_deg_mean, trans_l2_mean)
    """
    assert pred_01.shape == gt_01.shape and pred_01.shape[-2:] == (4, 4)
    R_pred = pred_01[..., :3, :3]
    t_pred = pred_01[..., :3, 3]
    R_gt = gt_01[..., :3, :3]
    t_gt = gt_01[..., :3, 3]

    R_rel = R_pred.transpose(-2, -1) @ R_gt
    t_rel = torch.einsum("...ij,...j->...i", R_pred.transpose(-2, -1), (t_gt - t_pred))
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    angle = acos_clamped((trace - 1.0) * 0.5)  # radians
    trans_norm = t_rel.norm(dim=-1)
    rot_deg_mean = (angle.mean() * 180.0 / torch.pi)
    trans_l2_mean = trans_norm.mean()
    return rot_deg_mean, trans_l2_mean


def error_auc(errors, thresholds=(5, 10, 20)):
    import numpy as np

    errors = [0] + sorted([float(e) for e in errors])
    recall = list(np.linspace(0, 1, len(errors)))
    aucs = {}
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[: last_index] + [recall[last_index - 1] if last_index > 0 else 0.0]
        x = errors[: last_index] + [thr]
        aucs[f"auc@{thr}"] = float(np.trapz(y, x) / thr)
    return aucs


def pose_err_components_deg(pred_01: torch.Tensor, gt_01: torch.Tensor):
    """
    Compute per-sample translation direction error (deg) and rotation error (deg)
    like MegaDepth-1500 metrics.
    Returns tensors of shape [B].
    """
    assert pred_01.shape == gt_01.shape and pred_01.shape[-2:] == (4, 4)
    R_pred = pred_01[..., :3, :3]
    t_pred = pred_01[..., :3, 3]
    R_gt = gt_01[..., :3, :3]
    t_gt = gt_01[..., :3, 3]

    # Rotation geodesic error
    R_rel = R_pred.transpose(-2, -1) @ R_gt
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    angle_rad = acos_clamped((trace - 1.0) * 0.5)
    R_err_deg = angle_rad * 180.0 / torch.pi

    # Translation direction angle error
    t_norm = (t_pred.norm(dim=-1) * t_gt.norm(dim=-1)).clamp_min(1e-8)
    cos_t = (t_pred * t_gt).sum(dim=-1) / t_norm
    t_err_deg = acos_clamped(cos_t) * 180.0 / torch.pi
    t_err_deg = torch.minimum(t_err_deg, 180.0 - t_err_deg)  # handle ambiguity

    return t_err_deg, R_err_deg


@hydra.main(version_base=None, config_path="../config", config_name="main_enhanced")
def main(cfg_dict: DictConfig):
    # Force test-like settings (no rendering), but we'll drive our own dataloader (MegaDepth).
    cfg_dict = OmegaConf.merge(cfg_dict, {
        "mode": "test",
        "test": {"compute_scores": True, "save_image": False, "save_video": False},
    })

    # Output dir
    out_root = Path("outputs") / cfg_dict.get("output_dir", "posenet_eval_megadepth")
    out_root.mkdir(parents=True, exist_ok=True)

    # Build typed cfg and set global
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model (encoder with enhancer). We'll invoke the enhancer directly.
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    from src.model.enhancer import get_enhancer
    enhancer, _ = get_enhancer(cfg.model.enhancer) if cfg.model.enhancer is not None else (None, None)
    encoder, _ = get_encoder(cfg.model.encoder, decoder, enhancer)

    # Load checkpoint weights into encoder (and enhancer inside it)
    ckpt_path = cfg.checkpointing.load
    if ckpt_path is not None and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu").get("state_dict", None)
        if sd is not None:
            # drop the first prefix segment to match encoder's state_dict keys
            stripped = {".".join(k.split(".")[1:]): v for k, v in sd.items()}
            encoder.load_state_dict(stripped, strict=False)

    encoder.to(device).eval()

    # MegaDepth1500 dataset (third_party)
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "third_party" / "accelerated_features"))
    from modules.eval.megadepth1500 import MegaDepth1500  # type: ignore

    dataset_dir = os.environ.get("MEGADEPTH_DIR", "/mnt/data/jun.wang03/megadepth")
    json_file = repo_root / "third_party" / "accelerated_features" / "assets" / "megadepth_1500.json"
    dataset = MegaDepth1500(json_file=str(json_file), root_dir=str(Path(dataset_dir) / "megadepth_test_1500"))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Decide pose input type
    pose_input = None
    try:
        pose_input = encoder.enhancer.pose_enhancer.cfg.input_data  # "image" | "feat" | "feat_backbone"
    except Exception:
        pose_input = "image"

    rot_list = []
    trans_list = []
    t_err_list = []
    R_err_list = []
    err_max_list = []

    with torch.inference_mode():
        for d in tqdm(loader, desc="Evaluating PoseNet on MegaDepth1500"):
            # Build context batch for enhancer
            img0 = d["image0"].to(device)  # [B,3,H,W]
            img1 = d["image1"].to(device)
            K0 = d["K0"].to(device)       # [B,3,3]
            K1 = d["K1"].to(device)
            T_0to1 = d["T_0to1"].to(device)  # [B,4,4]

            # Adjust intrinsics to the resized image sizes using scale (scale = [w/w_new, h/h_new]).
            # So sx = 1/scale_x, sy = 1/scale_y.
            if "scale0" in d and "scale1" in d:
                s0 = d["scale0"].to(device)  # [B,2]
                s1 = d["scale1"].to(device)  # [B,2]
                sx0, sy0 = (1.0 / s0[:, 0]), (1.0 / s0[:, 1])
                sx1, sy1 = (1.0 / s1[:, 0]), (1.0 / s1[:, 1])
                # fx, cx scale with width; fy, cy scale with height
                K0 = K0.clone()
                K1 = K1.clone()
                K0[:, 0, 0] = K0[:, 0, 0] * sx0
                K0[:, 1, 1] = K0[:, 1, 1] * sy0
                K0[:, 0, 2] = K0[:, 0, 2] * sx0
                K0[:, 1, 2] = K0[:, 1, 2] * sy0
                K1[:, 0, 0] = K1[:, 0, 0] * sx1
                K1[:, 1, 1] = K1[:, 1, 1] * sy1
                K1[:, 0, 2] = K1[:, 0, 2] * sx1
                K1[:, 1, 2] = K1[:, 1, 2] * sy1

            # Make both views the same spatial size by bottom-right padding (keeps pixel origin unchanged).
            _, _, h0, w0 = img0.shape
            _, _, h1, w1 = img1.shape
            Ht = max(h0, h1)
            Wt = max(w0, w1)
            if h0 != Ht or w0 != Wt:
                pad0 = (0, Wt - w0, 0, Ht - h0)  # (left, right, top, bottom)
                img0 = F.pad(img0, pad0, mode="constant", value=0.0)
            if h1 != Ht or w1 != Wt:
                pad1 = (0, Wt - w1, 0, Ht - h1)
                img1 = F.pad(img1, pad1, mode="constant", value=0.0)

            images = torch.stack([img0, img1], dim=1)  # [B,2,3,Ht,Wt]
            intrinsics = torch.stack([K0, K1], dim=1)  # [B,2,3,3]
            B, V, C, H, W = images.shape
            near = torch.full((B, V), 0.01, device=device, dtype=images.dtype)
            far = torch.full((B, V), 1000.0, device=device, dtype=images.dtype)

            batch = {
                "context": {
                    "image": images,
                    "intrinsics": intrinsics,
                    "near": near,
                    "far": far,
                }
            }

            # Apply enhancer batch shim (align + add norm_xy)
            if hasattr(encoder, "enhancer") and encoder.enhancer is not None:
                try:
                    shim = encoder.enhancer.get_batch_shim()
                    batch = shim(batch)
                except Exception:
                    pass

            # Compute features if needed
            features_list = ()
            if pose_input != "image":
                features_list = encoder.backbone(
                    batch["context"],
                    attn_splits=encoder.cfg.multiview_trans_attn_split,
                    return_cnn_features=True,
                    epipolar_kwargs=None,
                )

            # Run enhancer to get predicted SE3
            ctx, _ = encoder.enhancer(batch["context"], features_list) if hasattr(encoder, "enhancer") and encoder.enhancer is not None else (batch["context"], features_list)
            eo = ctx.get("_enhancer_outputs", {}) if isinstance(ctx, dict) else {}
            pred_01 = eo.get("pred_pose_0to1", None)  # [B,4,4]
            if pred_01 is None:
                # If enhancer not set, skip
                continue

            rot_deg, trans_l2 = pose_metrics_from_se3(pred_01, T_0to1)
            rot_list.append(float(rot_deg.item()))
            trans_list.append(float(trans_l2.item()))

            # MegaDepth-style components for MAA
            t_err_deg, R_err_deg = pose_err_components_deg(pred_01, T_0to1)
            # Average per batch (B==1 typically)
            t_val = float(t_err_deg.mean().item())
            R_val = float(R_err_deg.mean().item())
            t_err_list.append(t_val)
            R_err_list.append(R_val)
            err_max_list.append(max(t_val, R_val))

    # Compute MAA (AUC of max(t_err, R_err)) and mAcc@{5,10,20}
    from numpy import array as _np_array
    errors = err_max_list
    maa_auc = error_auc(errors) if errors else {}
    macc = {}
    if errors:
        import numpy as _np
        e = _np.array(errors)
        for thr in (5, 10, 20):
            macc[f"mAcc@{thr}"] = float((e <= thr).sum() / len(e))

    summary = {
        "num_samples": len(rot_list),
        "rot_deg_mean": float(sum(rot_list) / max(len(rot_list), 1)) if rot_list else None,
        "trans_l2_mean": float(sum(trans_list) / max(len(trans_list), 1)) if trans_list else None,
        "rot_auc": error_auc(rot_list) if rot_list else {},
        "maa_auc": maa_auc,
        "macc": macc,
    }

    out_path = out_root / "posenet_eval_megadepth.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved PoseNet eval summary to {out_path}")
    print(json.dumps(summary, indent=2))
    # Pretty print MAA like the original script
    if errors:
        print(f"auc / mAcc on {len(errors)} pairs")
        for k, v in maa_auc.items():
            print(f"{k} : {v*100:.1f}")
        for thr in (5, 10, 20):
            print(f"mAcc@{thr}: {macc[f'mAcc@{thr}']*100:.1f}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

