import os
import json
from pathlib import Path
from copy import deepcopy

import hydra
import torch
from omegaconf import DictConfig

from jaxtyping import install_import_hook

# Ensure typed modules are importable with beartype/jaxtyping
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.enhancer import get_enhancer

from pose_tools.eval_posenet_lib import evaluate_posenet_on_megadepth1500


@hydra.main(version_base=None, config_path="../config", config_name="main_enhanced")
def main(cfg_dict: DictConfig):
    # 输出目录（与训练评测写入 JSON 的路径保持一致风格）
    out_root = Path("outputs") / (cfg_dict.get("output_dir") or "posenet_eval_megadepth")
    out_root.mkdir(parents=True, exist_ok=True)

    # 构建 typed cfg 并设置全局 cfg
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建增强器，并强制为 pose 模式（与训练时一致）
    enh_cfg = deepcopy(cfg.model.enhancer)
    if hasattr(enh_cfg, "enhance_type"):
        enh_cfg.enhance_type = "pose"
    enhancer, _ = get_enhancer(enh_cfg)

    # 加载训练保存的权重到增强器本体（处理可能的 module. 前缀）
    ckpt_path = cfg.checkpointing.load
    if ckpt_path is not None and os.path.exists(ckpt_path):
        obj = torch.load(ckpt_path, map_location="cpu")
        sd = obj.get("state_dict", obj) if isinstance(obj, dict) else None
        if sd is not None:
            if any(k.startswith("module.") for k in sd.keys()):
                sd = {k[len("module."):]: v for k, v in sd.items()}
            try:
                enhancer.load_state_dict(sd, strict=True)
            except Exception as e:
                print(f"Warning: failed to load checkpoint into enhancer: {e}")

    enhancer.to(device).eval()

    # 直接复用库评测（与训练中内置评测完全一致）
    summary = evaluate_posenet_on_megadepth1500(
        enhancer=enhancer,
        device=device,
        output_dir=out_root,
        num_workers=4,
        batch_size=1,
        img_resize=(608, 800),
    )

    out_path = out_root / "posenet_eval_megadepth.json"
    print(f"Saved PoseNet eval summary to {out_path}")
    print(json.dumps(summary, indent=2))

    maa = summary.get("maa_auc", {}) or {}
    macc = summary.get("macc", {}) or {}
    if maa or macc:
        if maa:
            for k, v in maa.items():
                print(f"{k} : {v*100:.1f}")
        for thr in (5, 10, 20):
            key = f"mAcc@{thr}"
            if key in macc:
                print(f"{key}: {macc[key]*100:.1f}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

