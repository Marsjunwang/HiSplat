import os
import sys
from pathlib import Path
from dataclasses import dataclass
import torch.distributed as dist

import hydra
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel as DDP
from pose_tools.eval_posenet_lib import evaluate_posenet_on_megadepth1500

from jaxtyping import install_import_hook
import glob
import tqdm as _tqdm
from modules.dataset.megadepth.megadepth import MegaDepthDataset  # type: ignore
from torch.utils.tensorboard import SummaryWriter
import time
import logging
import contextlib

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


def make_dataloader(megadepth_root: str, batch_size: int, num_workers: int, 
    distributed: bool = False, rank: int = 0, world_size: int = 1
    ) -> DataLoader:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "third_party" / "accelerated_features"))

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

    dataset = ConcatDataset(datasets)
    sampler = None
    if distributed and world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
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

    # Setup distributed (single-node multi-GPU via torchrun)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1 and torch.cuda.is_available()
    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    # Output dir (only rank 0 sets up writer)
    rank = dist.get_rank() if is_distributed else 0
    is_main_process = (rank == 0)
    if is_main_process:
        out_root = Path("/mnt/data/jun.wang03/HiSplat/outputs") / (cfg_dict.get("output_dir") or "posenet_megadepth")
        logging.info(f"Output directory: {os.path.abspath(out_root)}")
        time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
        ckpt_dir = out_root / time_str / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(out_root / "logdir" / time_str) if is_main_process else None
    else:
        writer = None

    # Build typed cfg and set global
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Build PoseNet enhancer from config
    enh_cfg = deepcopy(cfg.model.enhancer)
    print(f"enh_cfg: {enh_cfg}")
    # Ensure pose-only for training
    if hasattr(enh_cfg, "enhance_type"):
        enh_cfg.enhance_type = "pose"
    enhancer, _ = get_enhancer(enh_cfg)
    losses = get_losses(cfg.loss)
    loss_fn = next((l for l in losses if getattr(l, "name", "") == "pose_relative"), None)
    enhancer.to(device)
    enhancer.train()
    if is_distributed:
        enhancer_ddp = DDP(enhancer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        train_module = enhancer_ddp
    else:
        train_module = enhancer

    # Optionally load checkpoint
    if cfg.checkpointing.load is not None and os.path.exists(cfg.checkpointing.load):
        sd = torch.load(cfg.checkpointing.load, map_location="cpu").get("state_dict", None)
        if sd is not None:
            stripped = {".".join(k.split(".")[1:]): v for k, v in sd.items()}
            target = train_module.module if hasattr(train_module, "module") else train_module
            target.load_state_dict(stripped, strict=False)

    # Optimizer
    # Read optimizer from top-level (preferred) or nested under posenet_train (fallback)
    optimizer_cfg = cfg_dict["posenet_train"].get("optimizer", None)
    
    opt_lr = optimizer_cfg.get("lr", cfg_dict["posenet_train"].get("lr", default.lr)) if optimizer_cfg is not None else cfg_dict["posenet_train"].get("lr", default.lr)
    opt_name = (optimizer_cfg.get("type", "adam") if optimizer_cfg is not None else "adam").lower()
    weight_decay = optimizer_cfg.get("weight_decay", default.weight_decay)
    parameters = filter(lambda p: p.requires_grad, train_module.parameters())
    if opt_name == "adamw":
        opt = optim.AdamW(parameters, lr=opt_lr, weight_decay=weight_decay)
    else:
        opt = optim.Adam(parameters, lr=opt_lr, weight_decay=weight_decay)
    # Scheduler: warmup + optional cosine from optimizer config
    steps_total = int(cfg_dict["posenet_train"]["steps"])
    warm_up_steps = 0
    use_cosine = False
    if optimizer_cfg is not None:
        warm_up_steps = int(optimizer_cfg.get("warm_up_steps", 0) or 0)
        use_cosine = bool(optimizer_cfg.get("cosine_lr", False))
    if use_cosine:
        base_lr = float(opt_lr)
        pct_start = max(0.0, min(0.9, (warm_up_steps / steps_total) if steps_total > 0 else 0.01))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=base_lr,
            total_steps=steps_total,
            pct_start=pct_start,
            cycle_momentum=False,
            anneal_strategy="cos",
        )
    else:
        if warm_up_steps > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=1 / max(1, warm_up_steps),
                end_factor=1.0,
                total_iters=warm_up_steps,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1)

    # AMP / Accumulation / EMA
    use_amp = bool(cfg_dict["posenet_train"].get("use_amp", True))
    accum_steps = int(cfg_dict["posenet_train"].get("accum_steps", 1))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ema_decay = float(cfg_dict["posenet_train"].get("ema_decay", 0.0) or 0.0)
    ema_model = None
    if ema_decay > 0.0:
        base_model = train_module.module if hasattr(train_module, "module") else train_module
        ema_model = deepcopy(base_model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)


    # Data
    megadepth_root = os.environ.get("MEGADEPTH_ROOT", "/mnt/data/jun.wang03/xfeat_data")
    loader = make_dataloader(
        megadepth_root,
        cfg_dict["posenet_train"]["batch_size"],
        cfg_dict["posenet_train"]["num_workers"],
        distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )

    global_step = 0
    micro_step = 0
    if is_main_process:
        pbar = tqdm(total=cfg_dict["posenet_train"]["steps"], desc="Train PoseNet(MegaDepth)")
    else:
        pbar = None
    data_iter = iter(loader)
    current_epoch = 0
    autocast_ctx = torch.cuda.amp.autocast \
        if torch.cuda.is_available() else contextlib.nullcontext
    while global_step < cfg_dict["posenet_train"]["steps"]:
        # Measure data loading time per iteration
        t_data_start = time.time()
        data_time = 0.0
        try:
            batch_md = next(data_iter)
            data_time = time.time() - t_data_start
        except StopIteration:
            # New epoch for distributed sampler to reshuffle
            if is_distributed and hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                current_epoch += 1
                loader.sampler.set_epoch(current_epoch)
            data_iter = iter(loader)
            batch_md = next(data_iter)
            data_time = time.time() - t_data_start

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

        # Optimizer zero_grad only at accumulation window start
        if (global_step == 0 and micro_step == 0) or (micro_step % accum_steps == 0):
            opt.zero_grad(set_to_none=True)

        # Forward
        if hasattr(enhancer, "get_batch_shim"):
            context = enhancer.get_batch_shim()(
                {"context": context})["context"]

        # Measure forward(inference) time
        t_fwd_start = time.time()
        with autocast_ctx(enabled=use_amp):
            ctx, _ = train_module(context, ())
            eo = ctx.get("_enhancer_outputs", {}) if isinstance(ctx, dict) else {}
            pred_01 = eo.get("pred_pose_0to1", None)
            if pred_01 is None:
                micro_step += 1
                continue

            gt_01 = batch_md["T_0to1"]
            output = DecoderOutput(color=images)
            output.pred_pose_0to1 = pred_01
            output.gt_pose_0to1 = gt_01
            loss, angle, trans = loss_fn.forward_posenet(
                output, None, None, global_step)
            fwd_time = time.time() - t_fwd_start

        # Backward with grad accumulation
        scaled_loss = loss / max(1, accum_steps)
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        step_now = ((micro_step + 1) % accum_steps) == 0
        if step_now:
            if cfg_dict["posenet_train"].get("grad_clip", 0.0) and cfg_dict["posenet_train"]["grad_clip"] > 0:
                if use_amp:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(train_module.parameters(), cfg_dict["posenet_train"]["grad_clip"]) 
            if use_amp:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            scheduler.step()

            # EMA update
            if ema_model is not None:
                with torch.no_grad():
                    msd = (train_module.module.state_dict() if hasattr(train_module, "module") else train_module.state_dict())
                    for k, v in ema_model.state_dict().items():
                        src = msd[k]
                        if v.dtype.is_floating_point:
                            v.mul_(ema_decay).add_(src, alpha=1.0 - ema_decay)
                        else:
                            v.copy_(src)

            global_step += 1
            if pbar is not None and is_main_process:
                pbar.set_description(f"Train PoseNet(MD) | step {global_step} | loss {loss.item():.4f} rot_deg {(angle.mean()*180/torch.pi).item():.2f} trans {trans.mean().item():.3f}")
                pbar.update(1)
                # Show timing (ms) in tqdm postfix
                try:
                    pbar.set_postfix({
                        "data_ms": f"{data_time * 1000.0:.1f}",
                        "fwd_ms": f"{fwd_time * 1000.0:.1f}",
                    })
                except Exception:
                    pass
            if writer is not None and is_main_process:
                writer.add_scalar("loss", loss.item(), global_step)
                writer.add_scalar("rot_deg", (angle.mean()*180/torch.pi).item(), global_step)
                writer.add_scalar("trans", trans.mean().item(), global_step)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                # Log timing to TensorBoard
                try:
                    writer.add_scalar("time/data_ms", float(data_time * 1000.0), global_step)
                    writer.add_scalar("time/fwd_ms", float(fwd_time * 1000.0), global_step)
                except Exception:
                    pass

            if global_step % cfg_dict["posenet_train"]["save_every"] == 0:
                if is_distributed:
                    dist.barrier()
                if is_main_process:
                    save_path = ckpt_dir / f"posenet_md_step_{global_step}.pth"
                    state_dict = train_module.module.state_dict() if hasattr(train_module, "module") else train_module.state_dict()
                    torch.save({"state_dict": state_dict, "global_step": global_step}, save_path)

                    if ema_model is not None:
                        ema_path = ckpt_dir / f"posenet_md_step_{global_step}_ema.pth"
                        torch.save({"state_dict": ema_model.state_dict(), "global_step": global_step}, ema_path)

                    # In-process evaluation without Hydra/CLI
                    base_name = (cfg_dict.get("output_dir") or "posenet_megadepth")
                    eval_output_dir_name = f"{base_name}_eval_step_{global_step}"
                    try:
                        eval_module = ema_model if ema_model is not None else train_module
                        summary = evaluate_posenet_on_megadepth1500(
                            enhancer=eval_module,
                            device=device,
                            output_dir=out_root / time_str/ "eval" / eval_output_dir_name,
                            num_workers=4,
                            batch_size=1,
                        )
                        if writer is not None and isinstance(summary, dict):
                            if (v := summary.get("rot_deg_mean")) is not None:
                                writer.add_scalar("eval/rot_deg_mean", float(v), global_step)
                            if (v := summary.get("trans_l2_mean")) is not None:
                                writer.add_scalar("eval/trans_l2_mean", float(v), global_step)
                            maa = summary.get("maa_auc", {}) or {}
                            for k, v in maa.items():
                                writer.add_scalar(f"eval/maa_{k}", float(v), global_step)
                            macc = summary.get("macc", {}) or {}
                            for k, v in macc.items():
                                writer.add_scalar(f"eval/{k}", float(v), global_step)
                    except Exception as e:
                        print(f"Evaluation failed at step {global_step}: {e}")
                if is_distributed:
                    dist.barrier()

        micro_step += 1

    # Final save
    if is_main_process:
        save_path = ckpt_dir / f"posenet_md_final.pth"
        state_dict = train_module.module.state_dict() if hasattr(train_module, "module") else train_module.state_dict()
        torch.save({"state_dict": state_dict, "global_step": global_step}, save_path)
        print(f"Saved final PoseNet checkpoint to {save_path}")
        if ema_model is not None:
            ema_path = ckpt_dir / f"posenet_md_final_ema.pth"
            torch.save({"state_dict": ema_model.state_dict(), "global_step": global_step}, ema_path)
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()