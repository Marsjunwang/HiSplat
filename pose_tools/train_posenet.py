import os
import sys
from pathlib import Path
from dataclasses import dataclass

import hydra
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from jaxtyping import install_import_hook


with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.enhancer import get_enhancer
    from src.loss import get_losses
    from src.model.decoder.decoder import DecoderOutput


@dataclass
class DefaultTrainCfg:
    lr: float = 3e-4
    weight_decay: float = 0.0
    steps: int = 160_000
    batch_size: int = 8
    num_workers: int = 4
    save_every: int = 2000
    grad_clip: float = 1.0


def acos_clamped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.acos(x.clamp(min=-1.0 + eps, max=1.0 - eps))


def se3_residual(pred_01: torch.Tensor, gt_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert pred_01.shape == gt_01.shape and pred_01.shape[-2:] == (4, 4)
    R_pred = pred_01[..., :3, :3]
    t_pred = pred_01[..., :3, 3]
    R_gt = gt_01[..., :3, :3]
    t_gt = gt_01[..., :3, 3]
    R_rel = R_pred.transpose(-2, -1) @ R_gt
    t_rel = torch.einsum("...ij,...j->...i", R_pred.transpose(-2, -1), (t_gt - t_pred))
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    angle = acos_clamped((trace - 1.0) * 0.5)
    trans_norm = t_rel.norm(dim=-1)
    return angle, trans_norm


def make_dataloader(megadepth_root: str, batch_size: int, num_workers: int) -> DataLoader:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "third_party" / "accelerated_features"))
    import glob
    import tqdm as _tqdm
    from modules.dataset.megadepth.megadepth import MegaDepthDataset  # type: ignore

    TRAIN_BASE_PATH = f"{megadepth_root}/train_data/megadepth_indices"
    TRAINVAL_DATA_SOURCE = f"{megadepth_root}/MegaDepth_v1"
    TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
    npz_paths = glob.glob(TRAIN_NPZ_ROOT + "/*.npz")

    # Concat multiple scenes; images are resized (800,608) in dataset, enabling batching.
    datasets = [
        MegaDepthDataset(
            root_dir=TRAVAL_DATA_SOURCE if (TRAVAL_DATA_SOURCE := TRAINVAL_DATA_SOURCE) else TRAINVAL_DATA_SOURCE,
            npz_path=path,
            mode="train",
            img_resize=(800, 608),
            df=32,
            img_padding=False,
            depth_padding=False,
            load_depth=False,
        )
        for path in _tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")
    ]

    from torch.utils.data import ConcatDataset
    dataset = ConcatDataset(datasets)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def adjust_intrinsics_with_scale(K: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # scale = [w/w_new, h/h_new] → sx=1/scale_x, sy=1/scale_y
    assert K.dim() == 3 and scale.dim() == 2 and K.shape[0] == scale.shape[0]
    sx, sy = (1.0 / scale[:, 0]), (1.0 / scale[:, 1])
    K = K.clone()
    K[:, 0, 0] *= sx
    K[:, 1, 1] *= sy
    K[:, 0, 2] *= sx
    K[:, 1, 2] *= sy
    return K

@hydra.main(version_base=None, config_path="../config", config_name="main_enhanced")
def main(cfg_dict: DictConfig):
    # Merge default train settings if not provided
    default = DefaultTrainCfg()
    cfg_dict = OmegaConf.merge(
        cfg_dict,
        {
            "mode": "train",
            "posenet_train": {
                "lr": default.lr,
                "weight_decay": default.weight_decay,
                "steps": default.steps,
                "batch_size": default.batch_size,
                "num_workers": default.num_workers,
                "save_every": default.save_every,
                "grad_clip": default.grad_clip,
            },
        },
    )

    # Output dir
    out_root = Path("outputs") / (cfg_dict.get("output_dir") or "posenet_megadepth")
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build typed cfg and set global
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build PoseNet enhancer from config
    from copy import deepcopy
    enh_cfg = deepcopy(cfg.model.enhancer)
    # Ensure pose-only for training
    if hasattr(enh_cfg, "enhance_type"):
        enh_cfg.enhance_type = "pose"
    enhancer, _ = get_enhancer(enh_cfg)
    losses = get_losses(cfg.loss)
    loss_fn = next((l for l in losses if getattr(l, "name", "") == "pose_relative"), None)
    enhancer.to(device)
    enhancer.train()

    # Optionally load checkpoint
    if cfg.checkpointing.load is not None and os.path.exists(cfg.checkpointing.load):
        sd = torch.load(cfg.checkpointing.load, map_location="cpu").get("state_dict", None)
        if sd is not None:
            stripped = {".".join(k.split(".")[1:]): v for k, v in sd.items()}
            enhancer.load_state_dict(stripped, strict=False)

    # Optimizer
    opt = optim.Adam(
        filter(lambda p: p.requires_grad, enhancer.parameters()),
        lr=cfg_dict["posenet_train"]["lr"],
        weight_decay=cfg_dict["posenet_train"]["weight_decay"],
    )

    # Loss weights
    # weight_rot = float(getattr(cfg.loss.pose_relative, "weight_rot", 1.0)) if hasattr(cfg, "loss") and hasattr(cfg, "loss") else 1.0
    # weight_trans = float(getattr(cfg.loss.pose_relative, "weight_trans", 1.0)) if hasattr(cfg, "loss") and hasattr(cfg, "loss") else 1.0

    # Data
    megadepth_root = os.environ.get("MEGADEPTH_ROOT", "/mnt/data/jun.wang03/xfeat_data")
    loader = make_dataloader(megadepth_root, cfg_dict["posenet_train"]["batch_size"], cfg_dict["posenet_train"]["num_workers"])

    global_step = 0
    pbar = tqdm(total=cfg_dict["posenet_train"]["steps"], desc="Train PoseNet(MegaDepth)")
    data_iter = iter(loader)

    while global_step < cfg_dict["posenet_train"]["steps"]:
        try:
            batch_md = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch_md = next(data_iter)

        # Move to device
        for k, v in list(batch_md.items()):
            if isinstance(v, torch.Tensor):
                batch_md[k] = v.to(device)

        # Build context for enhancer (B,2,...) and adjust intrinsics to resized images
        img0, img1 = batch_md["image0"], batch_md["image1"]  # [B,3,H,W]
        K0, K1 = batch_md["K0"], batch_md["K1"]             # [B,3,3]
        s0, s1 = batch_md.get("scale0", None), batch_md.get("scale1", None)
        if s0 is not None and s1 is not None:
            K0 = adjust_intrinsics_with_scale(K0, s0)
            K1 = adjust_intrinsics_with_scale(K1, s1)

        # If needed, pad to common size per-batch
        _, _, h0, w0 = img0.shape
        _, _, h1, w1 = img1.shape
        Ht, Wt = max(h0, h1), max(w0, w1)
        if h0 != Ht or w0 != Wt:
            img0 = F.pad(img0, (0, Wt - w0, 0, Ht - h0))
        if h1 != Ht or w1 != Wt:
            img1 = F.pad(img1, (0, Wt - w1, 0, Ht - h1))

        images = torch.stack([img0, img1], dim=1)             # [B,2,3,H,W]
        intrinsics = torch.stack([K0, K1], dim=1)              # [B,2,3,3]
        B, V, C, H, W = images.shape
        near = torch.full((B, V), 0.01, device=device, dtype=images.dtype)
        far = torch.full((B, V), 1000.0, device=device, dtype=images.dtype)

        context = {
            "image": images,
            "intrinsics": intrinsics,
            "near": near,
            "far": far,
        }

        enhancer.zero_grad(set_to_none=True)
        # Forward
        ctx, _ = enhancer(context, ())
        eo = ctx.get("_enhancer_outputs", {}) if isinstance(ctx, dict) else {}
        pred_01 = eo.get("pred_pose_0to1", None)
        if pred_01 is None:
            continue

        gt_01 = batch_md["T_0to1"] 
        output = DecoderOutput(color=images)# [B,4,4]
        output.pred_pose_0to1 = pred_01
        output.gt_pose_0to1 = gt_01
        loss, angle, trans = loss_fn.forward_posenet(output, None, None, global_step)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(enhancer.parameters(), cfg_dict["posenet_train"]["grad_clip"])
        opt.step()

        global_step += 1
        pbar.set_description(f"Train PoseNet(MD) | step {global_step} | loss {loss.item():.4f} rot_deg {(angle.mean()*180/torch.pi).item():.2f} trans {trans.mean().item():.3f}")
        pbar.update(1)

        if global_step % cfg_dict["posenet_train"]["save_every"] == 0:
            save_path = ckpt_dir / f"posenet_md_step_{global_step}.pth"
            torch.save({"state_dict": enhancer.state_dict(), "global_step": global_step}, save_path)

    # Final save
    save_path = ckpt_dir / f"posenet_md_final.pth"
    torch.save({"state_dict": enhancer.state_dict(), "global_step": global_step}, save_path)
    print(f"Saved final PoseNet checkpoint to {save_path}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()