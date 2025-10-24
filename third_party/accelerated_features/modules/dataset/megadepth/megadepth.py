"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    MegaDepth data handling was adapted from 
    LoFTR official code: https://github.com/zju3dv/LoFTR/blob/master/src/datasets/megadepth.py
"""

import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
from modules.dataset.megadepth.utils import read_megadepth_gray, read_megadepth_depth, fix_path_from_d2net
import numpy.random as rnd

import pdb, tqdm, os
import cv2


class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score = 0.3, #0.3,
                 max_overlap_score = 1.0, #1,
                 load_depth = True,
                 img_resize = (800,608), #or None
                 df=32,
                 img_padding=False,
                 depth_padding=True,
                 augment_fn=None,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]
        self.load_depth = load_depth
        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            min_overlap_score = 0
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score and pair_info[1] < max_overlap_score]

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None #and img_padding and depth_padding

        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)
        #pdb.set_trace()
        for idx in range(len(self.scene_info['image_paths'])):
            self.scene_info['image_paths'][idx] = fix_path_from_d2net(self.scene_info['image_paths'][idx])

        for idx in range(len(self.scene_info['depth_paths'])):
            self.scene_info['depth_paths'][idx] = fix_path_from_d2net(self.scene_info['depth_paths'][idx])


    def __len__(self):
        return len(self.pair_infos)

    def _homo_rot(self, image0, image1, K0, K1, T_0to1, return_all=False, save_path=None, alpha=0.5):
        """
        使用纯旋转单应矩阵, 将 image0 扭正到 image1 的视角并可视化.

        Args:
            image0: torch.Tensor[C,H,W] 或 numpy.ndarray[H,W,3/1]
            image1: torch.Tensor[C,H,W] 或 numpy.ndarray[H,W,3/1]
            K0: torch.Tensor[3,3] 或 numpy.ndarray[3,3]
            K1: torch.Tensor[3,3] 或 numpy.ndarray[3,3]
            T_0to1: torch.Tensor[4,4] 或 numpy.ndarray[4,4]
            return_all: 若为 True, 返回 (warped0, overlay, H)
            save_path: 若提供, 将拼贴图保存到该路径
            alpha: overlay 的混合权重

        Returns:
            warped0 或 (warped0, overlay, H) 依据 return_all.
        """

        def to_numpy_image(img):
            if torch.is_tensor(img):
                img_np = img.detach().cpu().numpy()
                if img_np.ndim == 3:  # C,H,W -> H,W,C
                    img_np = np.transpose(img_np, (1, 2, 0))
            else:
                img_np = img
            # 统一为 float32, [0,1]
            if img_np.dtype != np.float32:
                img_np = img_np.astype(np.float32)
            if img_np.max() > 1.5:  # 认为是 [0,255]
                img_np = img_np / 255.0
            return img_np

        def to_numpy_matrix(m):
            if torch.is_tensor(m):
                return m.detach().cpu().numpy()
            return m

        img0 = to_numpy_image(image0)
        img1 = to_numpy_image(image1)

        # 保证三通道便于可视化
        if img0.ndim == 2:
            img0 = np.repeat(img0[..., None], 3, axis=2)
        if img1.ndim == 2:
            img1 = np.repeat(img1[..., None], 3, axis=2)

        K0_np = to_numpy_matrix(K0)
        K1_np = to_numpy_matrix(K1)
        T_np = to_numpy_matrix(T_0to1)

        R = T_np[:3, :3]
        # 纯旋转单应矩阵: H = K1 * R * K0^{-1}
        H = K1_np @ R @ np.linalg.inv(K0_np)

        h, w = img1.shape[0], img1.shape[1]
        warped0 = cv2.warpPerspective(img0, H.astype(np.float64), (w, h), flags=cv2.INTER_LINEAR)

        # 叠加可视化
        overlay = np.clip(alpha * warped0 + (1.0 - alpha) * img1, 0.0, 1.0)

        # 拼贴: 原图0, 目标图1, 扭正后的0, overlay
        try:
            tile = np.concatenate([img0, img1, warped0, overlay], axis=1)
        except Exception:
            # 如尺寸不一致则只拼接 img1 与 warped0
            tile = np.concatenate([img1, warped0, overlay], axis=1)

        if save_path is not None:
            cv2.imwrite(save_path, (tile * 255.0).astype(np.uint8))

        if return_all:
            return warped0, overlay, H
        return warped0

    def visualize_homo_rot(self, image0, image1, K0, K1, T_0to1, save_path, alpha=0.5):
        """
        简便可视化入口: 保存 image0->image1 的单应扭正与叠加结果.
        """
        _ = self._homo_rot(image0, image1, K0, K1, T_0to1, return_all=False, save_path=save_path, alpha=alpha)

    @staticmethod
    def _adjust_intrinsics_with_scale_single(K: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """根据 resize 的 scale 调整单帧内参.
        scale = [w/w_new, h/h_new], 因此 fx' = fx/scale_x, fy' = fy/scale_y.
        """
        assert K.shape == (3, 3)
        assert scale.numel() == 2
        Kn = K.clone()
        sx = 1.0 / scale[0]
        sy = 1.0 / scale[1]
        Kn[0, 0] = Kn[0, 0] * sx
        Kn[1, 1] = Kn[1, 1] * sy
        Kn[0, 2] = Kn[0, 2] * sx
        Kn[1, 2] = Kn[1, 2] * sy
        return Kn

    @staticmethod
    def _normalize_intrinsics(K: torch.Tensor, img_shape: tuple[int, int]) -> torch.Tensor:
        """将像素内参按图像尺寸 (H,W) 归一化到 [0,1] 坐标系."""
        h, w = int(img_shape[0]), int(img_shape[1])
        Kn = K.clone()
        Kn[0] = Kn[0] / float(w)
        Kn[1] = Kn[1] / float(h)
        return Kn

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx % len(self)]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
        
        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0, mask0, scale0 = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1 = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        if self.load_depth:
            # read depth. shape: (h, w)
            if self.mode in ['train', 'val']:
                depth0 = read_megadepth_depth(
                    osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size, resize=self.img_resize)
                depth1 = read_megadepth_depth(
                    osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size, resize=self.img_resize)
            else:
                depth0 = depth1 = torch.tensor([])

            # read intrinsics of original size
            K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
            K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)

            # read and compute relative poses
            T0 = self.scene_info['poses'][idx0]
            T1 = self.scene_info['poses'][idx1]
            T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
            T_1to0 = T_0to1.inverse()

            # 调整内参到resize后尺寸，并提供归一化版本
            H0, W0 = int(image0.shape[-2]), int(image0.shape[-1])
            H1, W1 = int(image1.shape[-2]), int(image1.shape[-1])
            K0_resized = self._adjust_intrinsics_with_scale_single(K_0, scale0)
            K1_resized = self._adjust_intrinsics_with_scale_single(K_1, scale1)
            K0_norm = self._normalize_intrinsics(K0_resized, (H0, W0))
            K1_norm = self._normalize_intrinsics(K1_resized, (H1, W1))

            data = {
                'image0': image0,  # (1, h, w)
                'depth0': depth0,  # (h, w)
                'image1': image1,
                'depth1': depth1,
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'K0_resized': K0_resized,
                'K1_resized': K1_resized,
                'K0_norm': K0_norm,
                'K1_norm': K1_norm,
                'scale0': scale0,  # [scale_w, scale_h]
                'scale1': scale1,
                'dataset_name': 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            }

            # for LoFTR training
            if mask0 is not None:  # img_padding is True
                if self.coarse_scale:
                    [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                        scale_factor=self.coarse_scale,
                                                        mode='nearest',
                                                        recompute_scale_factor=False)[0].bool()
                data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        else:
            
            # read intrinsics of original size
            K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
            K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)

            # read and compute relative poses
            T0 = self.scene_info['poses'][idx0]
            T1 = self.scene_info['poses'][idx1]
            T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
            T_1to0 = T_0to1.inverse()

            # 调整内参到resize后尺寸，并提供归一化版本
            H0, W0 = int(image0.shape[-2]), int(image0.shape[-1])
            H1, W1 = int(image1.shape[-2]), int(image1.shape[-1])
            K0_resized = self._adjust_intrinsics_with_scale_single(K_0, scale0)
            K1_resized = self._adjust_intrinsics_with_scale_single(K_1, scale1)
            K0_norm = self._normalize_intrinsics(K0_resized, (H0, W0))
            K1_norm = self._normalize_intrinsics(K1_resized, (H1, W1))

            data = {
                'image0': image0,  # (1, h, w)
                'image1': image1,
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'K0_resized': K0_resized,
                'K1_resized': K1_resized,
                'K0_norm': K0_norm,
                'K1_norm': K1_norm,
                'scale0': scale0,  # [scale_w, scale_h]
                'scale1': scale1,
                'dataset_name': 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            }
        
        # 使用resize后的像素内参进行可视化单应变换
        # self.visualize_homo_rot(image0, image1, K0_resized, K1_resized, T_0to1, save_path=f'{self.scene_id}_{idx0}_{idx1}.png')
        # exit()
        return data