import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def acos_clamped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.acos(x.clamp(min=-1.0 + eps, max=1.0 - eps))


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


def pose_metrics_from_se3(pred_01: torch.Tensor, gt_01: torch.Tensor):
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


def pose_err_components_deg(pred_01: torch.Tensor, gt_01: torch.Tensor):
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
    t_err_deg = torch.minimum(t_err_deg, 180.0 - t_err_deg)

    return t_err_deg, R_err_deg


def _build_megadepth1500_loader(batch_size: int = 1, num_workers: int = 4) -> DataLoader:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "third_party" / "accelerated_features"))
    from modules.eval.megadepth1500 import MegaDepth1500  # type: ignore

    dataset_dir = os.environ.get("MEGADEPTH_DIR", "/mnt/data/jun.wang03/megadepth")
    json_file = repo_root / "third_party" / "accelerated_features" / "assets" / "megadepth_1500.json"
    dataset = MegaDepth1500(json_file=str(json_file), root_dir=str(Path(dataset_dir) / "megadepth_test_1500"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader


def evaluate_posenet_on_megadepth1500(
    enhancer: torch.nn.Module,
    device: torch.device,
    output_dir: Path | str,
    *,
    num_workers: int = 4,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate a PoseNet-style enhancer directly (no Hydra/config) on MegaDepth1500.

    - enhancer: an initialized model; may be DDP-wrapped
    - device: device to run evaluation on
    - output_dir: directory to write JSON summary
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = _build_megadepth1500_loader(batch_size=batch_size, num_workers=num_workers)

    enhancer_module = enhancer.module if hasattr(enhancer, "module") else enhancer
    was_training = enhancer_module.training
    enhancer_module.eval()
    enhancer_module.to(device)

    rot_list = []
    trans_list = []
    t_err_list = []
    R_err_list = []
    err_max_list = []

    with torch.inference_mode():
        for d in tqdm(loader, desc="Evaluating PoseNet on MegaDepth1500"):
            img0 = d["image0"].to(device)
            img1 = d["image1"].to(device)
            K0 = d["K0"].to(device)
            K1 = d["K1"].to(device)
            T_0to1 = d["T_0to1"].to(device)

            if "scale0" in d and "scale1" in d:
                s0 = d["scale0"].to(device)
                s1 = d["scale1"].to(device)
                sx0, sy0 = (1.0 / s0[:, 0]), (1.0 / s0[:, 1])
                sx1, sy1 = (1.0 / s1[:, 0]), (1.0 / s1[:, 1])
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

            _, _, h0, w0 = img0.shape
            _, _, h1, w1 = img1.shape
            Ht = max(h0, h1)
            Wt = max(w0, w1)
            if h0 != Ht or w0 != Wt:
                pad0 = (0, Wt - w0, 0, Ht - h0)
                img0 = F.pad(img0, pad0, mode="constant", value=0.0)
            if h1 != Ht or w1 != Wt:
                pad1 = (0, Wt - w1, 0, Ht - h1)
                img1 = F.pad(img1, pad1, mode="constant", value=0.0)

            images = torch.stack([img0, img1], dim=1)
            intrinsics = torch.stack([K0, K1], dim=1)
            B, V, C, H, W = images.shape
            near = torch.full((B, V), 0.01, device=device, dtype=images.dtype)
            far = torch.full((B, V), 1000.0, device=device, dtype=images.dtype)

            context = {
                "image": images,
                "intrinsics": intrinsics,
                "near": near,
                "far": far,
            }
            if hasattr(enhancer_module, "get_batch_shim"):
                context = enhancer_module.get_batch_shim()(
                    {"context": context})["context"]
            ctx, _ = enhancer_module(context, ())
            eo = ctx.get("_enhancer_outputs", {}) if isinstance(ctx, dict) else {}
            pred_01 = eo.get("pred_pose_0to1", None)
            if pred_01 is None:
                continue

            rot_deg, trans_l2 = pose_metrics_from_se3(pred_01, T_0to1)
            rot_list.append(float(rot_deg.item()))
            trans_list.append(float(trans_l2.item()))

            t_err_deg, R_err_deg = pose_err_components_deg(pred_01, T_0to1)
            t_val = float(t_err_deg.mean().item())
            R_val = float(R_err_deg.mean().item())
            t_err_list.append(t_val)
            R_err_list.append(R_val)
            err_max_list.append(max(t_val, R_val))

    summary = {
        "num_samples": len(rot_list),
        "rot_deg_mean": float(sum(rot_list) / max(len(rot_list), 1)) if rot_list else None,
        "trans_l2_mean": float(sum(trans_list) / max(len(trans_list), 1)) if trans_list else None,
        "rot_auc": error_auc(rot_list) if rot_list else {},
        "maa_auc": error_auc(err_max_list) if err_max_list else {},
        "macc": {},
    }
    if err_max_list:
        import numpy as _np
        e = _np.array(err_max_list)
        for thr in (5, 10, 20):
            summary["macc"][f"mAcc@{thr}"] = float((e <= thr).sum() / len(e))

    out_path = Path(output_dir) / "posenet_eval_megadepth.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    if was_training:
        enhancer_module.train()

    return summary

