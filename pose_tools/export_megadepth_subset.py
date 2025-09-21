#!/usr/bin/env python3
"""
Export a small offline subset of a MegaDepth scene for debugging.
- Samples N pairs from an existing scene npz
- Copies required images/depths into a new root, preserving relative paths
- Writes a remapped npz referencing only the sampled items

Usage example:
  python tools/export_megadepth_subset.py \
    --root /path/to/megadepth/root \
    --npz /path/to/scene/0000.npz \
    --out-root /tmp/megadepth_subset/root \
    --out-npz /tmp/megadepth_subset/0000_subset.npz \
    --num-pairs 50 --seed 42
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import random
import shutil
from typing import Dict, List, Tuple

import numpy as np
import pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a MegaDepth subset for offline debugging")
    parser.add_argument("--root", default='/mnt/data/jun.wang03/xfeat_data/MegaDepth_v1', help="Original MegaDepth root directory (contains phoenix, etc.)")
    parser.add_argument("--npz", default='/mnt/data/jun.wang03/xfeat_data/train_data/megadepth_indices/scene_info_0.1_0.7/0000_0.3_0.5.npz', help="Path to scene npz (contains pair_infos and arrays)")
    parser.add_argument("--out-root", default="/mnt/data/jun.wang03/xfeat_data_mini/MegaDepth_v1", help="Output root to store copied files (will preserve relative paths)")
    parser.add_argument("--out-npz", default="/mnt/data/jun.wang03/xfeat_data_mini/train_data/megadepth_indices/scene_info_0.1_0.7/0000_0.1_0.3.npz", help="Output npz path for the subset scene")
    parser.add_argument("--num-pairs", type=int, default=10, help="Number of pairs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--no-copy", action="store_true", help="Do not copy files, only write subset npz")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    if path and not osp.exists(path):
        os.makedirs(path, exist_ok=True)


def safe_copy(src: str, dst: str) -> None:
    ensure_dir(osp.dirname(dst))
    if not osp.exists(src):
        raise FileNotFoundError(f"Source not found: {src}")
    if osp.exists(dst):
        # Do not overwrite unnecessarily
        return
    shutil.copy2(src, dst)


def to_str_list(arr) -> List[str]:
    # Handle arrays possibly containing bytes objects
    out: List[str] = []
    for v in arr:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out

def fix_path_from_d2net(path):
    if not path:
        return None

    path = path.replace('Undistorted_SfM/', '')
    path = path.replace('images', 'dense0/imgs')
    path = path.replace('phoenix/S6/zl548/MegaDepth_v1/', '')

    return path


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)

    print(f"[INFO] Loading scene: {args.npz}")
    scene = np.load(args.npz, allow_pickle=True)

    # Required fields expected by the dataset
    try:
        pair_infos = list(scene["pair_infos"])  # ((idx0, idx1), overlap_score, central_matches)
        image_paths = to_str_list(scene["image_paths"])  # list of relative paths
        depth_paths = to_str_list(scene["depth_paths"])  # list of relative paths
        intrinsics = np.array(scene["intrinsics"])      # (N, 3, 3)
        poses = np.array(scene["poses"])                # (N, 4, 4) or similar
    except KeyError as e:
        missing = str(e)
        raise KeyError(f"Missing key in npz: {missing}. Required: pair_infos, image_paths, depth_paths, intrinsics, poses")

    total_pairs = len(pair_infos)
    if total_pairs == 0:
        raise ValueError("pair_infos is empty; nothing to sample.")

    num_pairs = min(args.num_pairs, total_pairs)
    print(f"[INFO] Sampling {num_pairs} / {total_pairs} pairs (seed={args.seed})")

    sampled_pairs = rng.sample(pair_infos, k=num_pairs) if num_pairs < total_pairs else pair_infos

    # Determine the set of image indices used by sampled pairs
    used_indices_set = set()
    for entry in sampled_pairs:
        (idx0, idx1), _overlap, _central = entry
        used_indices_set.add(int(idx0))
        used_indices_set.add(int(idx1))

    used_indices = sorted(used_indices_set)

    # Build remapping from old index -> new index in subset arrays
    old_to_new: Dict[int, int] = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}

    # Create subset arrays
    new_image_paths: List[str] = [image_paths[i] for i in used_indices]
    new_depth_paths: List[str] = [depth_paths[i] for i in used_indices]
    new_intrinsics = np.stack([intrinsics[i] for i in used_indices], axis=0)
    new_poses = np.stack([poses[i] for i in used_indices], axis=0)

    # Remap pair indices
    new_pair_infos: List[Tuple] = []
    for entry in sampled_pairs:
        (idx0, idx1), overlap_score, central_matches = entry
        new_idx0 = old_to_new[int(idx0)]
        new_idx1 = old_to_new[int(idx1)]
        new_pair_infos.append(((new_idx0, new_idx1), overlap_score, central_matches))

    # Optionally copy required files into out-root, preserving relative paths
    if not args.no_copy:
        print(f"[INFO] Copying files to: {args.out_root}")
        for rel_path in new_image_paths:
            rel_path = fix_path_from_d2net(rel_path)
            src = osp.join(args.root, rel_path)
            dst = osp.join(args.out_root, rel_path)
            safe_copy(src, dst)
        for rel_path in new_depth_paths:
            # Some pipelines may have empty strings for depth when not used
            rel_path = fix_path_from_d2net(rel_path)
            if rel_path:
                src = osp.join(args.root, rel_path)
                dst = osp.join(args.out_root, rel_path)
                safe_copy(src, dst)

    # Write subset npz
    ensure_dir(osp.dirname(args.out_npz))
    # np.savez(
    #     args.out_npz,
    #     pair_infos=np.asarray(new_pair_infos, dtype=object),
    #     image_paths=np.asarray(new_image_paths, dtype=object),
    #     depth_paths=np.asarray(new_depth_paths, dtype=object),
    #     intrinsics=np.asarray(new_intrinsics, dtype=np.float32),
    #     poses=np.asarray(new_poses, dtype=np.float32),
    # )
    scene_dict = {
    "pair_infos": np.asarray(new_pair_infos, dtype=object),   # 或者直接 list 也可
    "image_paths": np.asarray(new_image_paths, dtype=object),
    "depth_paths": np.asarray(new_depth_paths, dtype=object),
    "intrinsics": np.asarray(new_intrinsics, dtype=np.float32),
    "poses": np.asarray(new_poses, dtype=np.float32),
    }

    # 保存为 pickled dict（保持读取端 np.load 返回 dict 而非 NpzFile/0D 对象数组）
    with open(args.out_npz, "wb") as f:
        pickle.dump(scene_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[OK] Subset written:")
    print(f"     out_root: {args.out_root}")
    print(f"     out_npz : {args.out_npz}")
    print(f"     images  : {len(new_image_paths)} unique")
    print(f"     pairs   : {len(new_pair_infos)}")


if __name__ == "__main__":
    main()