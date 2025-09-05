from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossPoseRelativeCfg:
    weight_rot: float
    weight_trans: float


@dataclass
class LossPoseRelativeCfgWrapper:
    pose_relative: LossPoseRelativeCfg


class LossPoseRelative(Loss[LossPoseRelativeCfg, LossPoseRelativeCfgWrapper]):
    @staticmethod
    def _se3_residual(pred_01: Tensor, gt_01: Tensor) -> tuple[Tensor, Tensor]:
        """Compute geodesic rotation error (radians) and translation L2 for SE(3) delta.

        Numerically stable variant without explicit 4x4 inverse:
        R_rel = R_pred^T R_gt, t_rel = R_pred^T (t_gt - t_pred)
        Returns (angle, trans_norm) per batch.
        Shapes: pred_01, gt_01: [B, 4, 4]
        """
        assert pred_01.shape == gt_01.shape and pred_01.shape[-2:] == (4, 4)
        R_pred = pred_01[..., :3, :3]
        t_pred = pred_01[..., :3, 3]
        R_gt = gt_01[..., :3, :3]
        t_gt = gt_01[..., :3, 3]

        R_rel = R_pred.transpose(-2, -1) @ R_gt
        t_rel = torch.einsum("...ij,...j->...i", R_pred.transpose(-2, -1), (t_gt - t_pred))

        # Clamp trace to valid range for numerical stability
        trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        eps = 1e-6
        cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0 + eps, max=1.0 - eps)
        angle = torch.acos(cos_theta)
        trans_norm = t_rel.norm(dim=-1)
        return angle, trans_norm

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        pred_01 = getattr(prediction, "pred_pose_0to1", None)
        gt_01 = getattr(prediction, "gt_pose_0to1", None)
        if pred_01 is None or gt_01 is None:
            device = prediction.color.device
            return torch.zeros((), device=device)
        angle, trans = self._se3_residual(pred_01, gt_01)
        loss = self.cfg.weight_rot * angle.mean() + self.cfg.weight_trans * trans.mean()
        # Store diagnostics for logging (similar to file_context_0)
        self.last_rot_deg_mean = angle.mean() * 180.0 / torch.pi
        self.last_trans_mean = trans.mean()
        return loss

    def dynamic_forward(
        self,
        prediction: DecoderOutput,
        gt_image: Tensor,
        global_step: int,
        weight: float | None = None,
    ) -> Float[Tensor, ""]:
        # Not applicable for pose supervision; return zero to satisfy interface.
        return torch.zeros((), device=gt_image.device)

