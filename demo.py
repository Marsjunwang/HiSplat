# A simple demo to validate the installation and show the effectiveness of HiSplat
import os
from pathlib import Path

import hydra
import torch
import wandb
from colorama import Fore
from einops import rearrange, repeat
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torchvision.io import write_video

# Configure beartype and jaxtyping.
from src.misc.image_io import save_batch_images
from src.visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def trajectory_fn(t, batch):
    extrinsics = interpolate_extrinsics(
        batch["context"]["extrinsics"][0, 0],
        batch["context"]["extrinsics"][0, 1],
        t,
    )
    intrinsics = interpolate_intrinsics(
        batch["context"]["intrinsics"][0, 0],
        batch["context"]["intrinsics"][0, 1],
        t,
    )
    return extrinsics[None], intrinsics[None]


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def align_world_to_first_context_camera(example):
    """Align world frame to the first context camera frame.

    Assumes `example["context"]["extrinsics"]` and `example["target"]["extrinsics"]`
    are camera-to-world (c2w) 4x4 matrices with shape [batch, views, 4, 4].

    After alignment, the first context camera will have identity extrinsics, and
    all other cameras (including targets) will be transformed accordingly so that
    their extrinsics are expressed in this new world frame.
    """
    if "context" not in example or "extrinsics" not in example["context"]:
        return example

    c2w_first = example["context"]["extrinsics"][:, 0]  # [B, 4, 4]
    w2c_first = c2w_first.inverse()  # [B, 4, 4]

    def _left_apply_w2c(views):
        if views is None:
            return
        if "extrinsics" in views:
            # [B, 1, 4, 4] @ [B, V, 4, 4] -> [B, V, 4, 4]
            views["extrinsics"] = w2c_first[:, None] @ views["extrinsics"]

    _left_apply_w2c(example.get("context"))
    _left_apply_w2c(example.get("target"))
    return example


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="main",
)
@torch.no_grad()
def generate_video(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    if cfg_dict.output_dir is None:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    else:  # for resuming
        output_dir = (
            Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]).parents[1] / cfg_dict.output_dir
        )
        os.makedirs(output_dir, exist_ok=True)
    # Set the test output path now that output_dir is resolved
    cfg_dict["test"]["output_path"] = str(output_dir / "test")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({"id": cfg_dict.wandb.id, "resume": "must"})
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        if wandb.run is not None:
            wandb.run.log_code("src")
    elif cfg_dict.use_tensorboard is not None:
        tensorboard_dir = output_dir / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True, parents=True)
        logger = TensorBoardLogger(save_dir=output_dir)
        callbacks.append(LearningRateMonitor("step", True))
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",  # save the lastest k ckpt, can do offline test later
        )
    )
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = "_"

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    # Resolve accelerator/devices for CPU/GPU environments
    resolved_accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_devices = cfg.device
    if resolved_accelerator == "cpu":
        if isinstance(resolved_devices, str) and resolved_devices == "auto":
            resolved_devices = 1
        elif isinstance(resolved_devices, int):
            resolved_devices = max(1, resolved_devices)
        else:
            resolved_devices = 1
    trainer = Trainer(
        max_epochs=-1,
        accelerator=resolved_accelerator,
        logger=logger,
        devices=resolved_devices,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    )
    print(f"GPU number is {torch.cuda.device_count()}")
    torch.manual_seed(46 + trainer.global_rank)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder, decoder)

    if checkpoint_path is not None:
        ckpt = (
            torch.load(checkpoint_path)["state_dict"]
            if torch.cuda.is_available()
            else torch.load(checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        )
        ckpt = {".".join(k.split(".")[1:]): v for k, v in ckpt.items()}
        encoder.load_state_dict(ckpt)
    # choose the task or method
    run_wrapper = ModelWrapper

    model_wrapper = run_wrapper(
        cfg.optimizer, cfg.test, cfg.train, encoder, encoder_visualizer, decoder, get_losses(cfg.loss), step_tracker
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper = model_wrapper.eval().to(device)
    """ Obtain data"""
    example = (
        torch.load(os.path.join("demo", "demo_example.tar"))
        if torch.cuda.is_available()
        else torch.load(os.path.join("demo", "demo_example.tar"), map_location=torch.device("cpu"))
    )
    print(f"Obtain context images and camera poses from {os.path.join('demo', 'demo_example.tar')}!")
    # Move example tensors to the correct device
    for k in ["image", "extrinsics", "intrinsics", "near", "far"]:
        if k in example["context"]:
            example["context"][k] = example["context"][k].to(device)
    # Targets used below for rendering; move needed keys to device
    if "target" in example:
        for k in ["extrinsics", "intrinsics", "image", "near", "far"]:
            if k in example["target"] and hasattr(example["target"][k], "to"):
                example["target"][k] = example["target"][k].to(device)

    # Align world frame to the first context camera frame so all poses are relative to it
    example = align_world_to_first_context_camera(example)
    # Run the model and get gaussians
    gaussian_dict, result_dict = model_wrapper.encoder(example["context"], 0, False, scene_names=[example["scene"]])
    gaussians = gaussian_dict[f"stage2"]["gaussians"]
    print("Get guassians successfully!")
    extrinsics, intrinsics = (
        torch.cat(
            [
                example["context"]["extrinsics"][:, :1],
                example["target"]["extrinsics"],
                example["context"]["extrinsics"][:, 1:2],
            ],
            dim=1,
        ),
        torch.cat(
            [
                example["context"]["intrinsics"][:, :1],
                example["target"]["intrinsics"],
                example["context"]["intrinsics"][:, 1:2],
            ],
            dim=1,
        ),
    )
    num_frames = len(extrinsics[0])

    print(f"Begin to render the RGB video, num_frames={num_frames}...")

    _, _, _, h, w = example["context"]["image"].shape

    # TODO: Interpolate near and far planes?
    near = repeat(example["context"]["near"][:, 0], "b -> b v", v=num_frames)
    far = repeat(example["context"]["far"][:, 0], "b -> b v", v=num_frames)
    batch_size = 64
    if extrinsics.shape[1] <= batch_size:
        output_prob = model_wrapper.decoder.forward(gaussians, extrinsics, intrinsics, near, far, (h, w))
        video = output_prob.color[0]
    else:
        batch_list = list(range(0, extrinsics.shape[1], batch_size))
        if batch_list[-1] != (extrinsics.shape[1] - 1):
            batch_list += [extrinsics.shape[1] - 1]
        for k in range(len(batch_list)):
            if k == len(batch_list) - 1:
                break
            output_prob = model_wrapper.decoder.forward(
                gaussians,
                extrinsics[:, batch_list[k] : batch_list[k + 1]],
                intrinsics[:, batch_list[k] : batch_list[k + 1]],
                near[:, batch_list[k] : batch_list[k + 1]],
                far[:, batch_list[k] : batch_list[k + 1]],
                (h, w),
            )
            if k == 0:
                video = output_prob.color[0].cpu()
            else:
                video = torch.cat([video, output_prob.color[0].cpu()], dim=0)
    video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu()
    video = rearrange(video, "t c h w -> t h w c")
    fps = 24
    os.makedirs(os.path.join("demo", "output"), exist_ok=True)
    # write_video(os.path.join("demo", "output", "video.mp4"), video, fps=fps)
    import imageio
    import numpy as np

    # video: numpy array of shape (num_frames, height, width, 3), dtype=uint8
    video = video.numpy()
    with imageio.get_writer('demo/output/video.mp4', fps=fps) as writer:
        for frame in video:
            writer.append_data(frame)
    save_batch_images(example["context"]["image"][0], os.path.join("demo", "output", "context_img.png"))
    print(cyan(f"Saving output videos and context images to {os.path.join('demo', 'output')}."))


if __name__ == "__main__":
    generate_video()
