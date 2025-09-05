import torch


def align_world_to_view0(extrinsics: torch.Tensor) -> torch.Tensor:
    """Align world frame to the first view (view 0).

    Assumes extrinsics are camera-to-world (c2w) with shape [B, V, 4, 4].
    Returns new extrinsics where view 0 becomes identity and others are expressed
    in this coordinate frame.
    """
    assert extrinsics.dim() == 4 and extrinsics.shape[-2:] == (4, 4), "extrinsics must be [B, V, 4, 4]"
    base = extrinsics[:, 0]  # [B, 4, 4]
    base_inv = base.inverse()
    return base_inv[:, None] @ extrinsics


def relative_pose_0_to_1(extrinsics: torch.Tensor) -> torch.Tensor:
    """Compute relative pose from view 0 to view 1 in homogeneous 4x4.

    Given c2w matrices E0, E1, compute T_{0->1} = E1^{-1} @ E0.
    Shape: input [B, V, 4, 4] -> output [B, 4, 4].
    """
    assert extrinsics.dim() == 4 and extrinsics.shape[-2:] == (4, 4) and extrinsics.shape[1] >= 2
    e0 = extrinsics[:, 0]
    e1 = extrinsics[:, 1]
    return e1.inverse() @ e0


def split_pred_relative_two_directions(pred_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split predicted relative poses into 0->1 and 1->0 if provided.

    Accepts shapes:
    - [B, 2, 1, 4, 4] or [B, 2, 4, 4]: returns (pred_0to1, pred_1to0)
    - [B, 1, 1, 4, 4] or [B, 1, 4, 4]: returns (pred, pred.inverse())
    - [B, 4, 4]: returns (pred, pred.inverse())
    """
    M = pred_pose
    if M.dim() == 5 and M.shape[-2:] == (4, 4):
        # [B, N, 1, 4, 4]
        M = M.squeeze(-3)
    if M.dim() == 4 and M.shape[-2:] == (4, 4):
        # [B, N, 4, 4] or [B, 4, 4]
        if M.shape[1] == 2:
            return M[:, 0], M[:, 1]
        if M.shape[1] == 1:
            m01 = M[:, 0]
            return m01, m01.inverse()
        if M.shape[0] >= 1 and M.shape[1] == 4:
            # Ambiguous case; treat as [4,4]
            m01 = M
            return m01, m01.inverse()
    if M.dim() == 3 and M.shape[-2:] == (4, 4):
        # [B, 4, 4]
        m01 = M
        return m01, m01.inverse()
    raise ValueError(f"Unsupported pred_pose shape: {tuple(pred_pose.shape)}")

