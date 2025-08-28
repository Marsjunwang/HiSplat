import torch
from kornia.geometry.epipolar import (normalize_points, 
    normalize_transformation, motion_from_essential, 
    motion_from_essential_choose_solution, 
    triangulate_points, 
    symmetrical_epipolar_distance)
from kornia.geometry.epipolar.projection import depth_from_point

def normalize(kpts, intr):
    n_kpts = torch.zeros_like(kpts)
    fx, fy, cx, cy = intr[..., 0, 0], intr[..., 1, 1], intr[..., 0, 2], intr[..., 1, 2]
    n_kpts[..., 0] = (kpts[..., 0] - cx.unsqueeze(-1)) / fx.unsqueeze(-1)
    n_kpts[..., 1] = (kpts[..., 1] - cy.unsqueeze(-1)) / fy.unsqueeze(-1)
    return n_kpts

def weighted_eight_point_single(x0: torch.Tensor, 
                                x1: torch.Tensor, 
                                w: torch.Tensor) -> torch.Tensor:
    """
    Weighted eight-point (DLT) for a single batch: 
    x0,x1 [M,2], w [M] → E [3,3].
    """
    M = x0.shape[0]
    ones = torch.ones(M, 1, device=x0.device, dtype=x0.dtype)
    X = torch.cat([x0, ones], dim=-1)  # [M,3]
    Xp = torch.cat([x1, ones], dim=-1)
    A = torch.cat([
        Xp[:, 0:1] * X[:, 0:1], Xp[:, 0:1] * X[:, 1:2], Xp[:, 0:1],
        Xp[:, 1:2] * X[:, 0:1], Xp[:, 1:2] * X[:, 1:2], Xp[:, 1:2],
        X[:, 0:1],             X[:, 1:2],             ones,
    ], dim=-1)  # [M,9]
    Aw = A * torch.sqrt(w).unsqueeze(-1)
    # Solve Aw v = 0 via SVD
    _, _, Vh = torch.linalg.svd(Aw, full_matrices=False)
    Fm = Vh[-1].view(3, 3)
    # Enforce rank-2
    Uf, Sf, Vhf = torch.linalg.svd(Fm)
    Sf = Sf * Sf.new_tensor([1., 1., 0.])
    E = Uf @ torch.diag(Sf) @ Vhf
    return E

def decompose_E_single(E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    U, _, Vt = torch.linalg.svd(E)
    W = E.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R = U @ W @ Vt
    t = Vt[2] / Vt[2, 2]
    return R, t

def find_fundamental(points1: torch.Tensor, 
                     points2: torch.Tensor, 
                     weights: torch.Tensor) -> torch.Tensor:
    r"""Compute the fundamental matrix using the DLT formulation.

    The linear system is solved by using the Weighted Least Squares Solution 
    for the 8 Points algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3, 3)`.

    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape, points2.shape)
    if not (len(weights.shape) == 2 and weights.shape[1] == points1.shape[1]):
        raise AssertionError(weights.shape)

    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # Bx1xN

    ones = torch.ones_like(x1)

    # build equations system and solve DLT
    # https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-09-TwoViews-2.pdf
    # [x * x', x * y', x, y * x', y * y', y, x', y', 1]

    X = torch.cat([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], dim=-1)  # BxNx9

    # apply the weights to the linear system
    w_diag = torch.diag_embed(weights)
    X = w_diag @ X

    # compute eigevectors and retrieve the one with the smallest eigenvalue
    _, _, V = torch.svd(X)
    F_mat = V[..., -1].view(-1, 3, 3)

    # reconstruct and force the matrix to have rank2
    U, S, V = torch.svd(F_mat)
    rank_mask = torch.tensor([1.0, 1.0, 0.0], device=F_mat.device, dtype=F_mat.dtype)

    F_projected = U @ (torch.diag_embed(S * rank_mask) @ V.transpose(-2, -1))
    F_est = transform2.transpose(-2, -1) @ (F_projected @ transform1)

    return normalize_transformation(F_est)

def estimate_relative_pose_w8pt(kpts0, kpts1, intr0, intr1, confidence, t_scale, 
                                choose_closest=False, T_021=None, 
                                determine_inliers=False):
    if kpts0.shape[1] < 8:
        return None, None
    sum_conf = confidence.sum(dim=1, keepdim=True) + 1e-6
    confidence = confidence / sum_conf
    kpts0_norm = normalize(kpts0, intr0)
    kpts1_norm = normalize(kpts1, intr1)
    dev = intr0.device
    bs = intr0.shape[0]
    intr = torch.eye(3, device=dev).unsqueeze(0)
    Fs = find_fundamental(kpts0_norm, kpts1_norm, confidence.squeeze(-1))
    if choose_closest:
        Rs, ts = motion_from_essential(Fs)
        min_err = torch.full((bs,), 1e6, device=dev)
        min_err_pred_T021 = torch.eye(4, device=dev).unsqueeze(0).repeat(bs, 1, 1)
        for R, t in zip(Rs.permute(1, 0, 2, 3), ts.permute(1, 0, 2, 3)):
            pred_T021 = torch.eye(4, device=dev).unsqueeze(0).repeat(bs, 1, 1)
            pred_T021[:, :3, :3] = R
            # Avoid chained in-place on a view: compute once and assign
            t_scaled = t.squeeze(-1) * t_scale
            pred_T021[:, :3, 3] = t_scaled

            curr_err = compute_rotation_error(pred_T021, T_021, reduce=False) \
                + compute_translation_error_as_angle(
                    pred_T021, T_021, reduce=False)
            update_mask = curr_err < min_err
            min_err[update_mask] = curr_err[update_mask]
            min_err_pred_T021[update_mask] = pred_T021[update_mask]
    else:
        R, t, pts = motion_from_essential_choose_solution(Fs, intr, intr, kpts0_norm, kpts1_norm, mask=None)
        min_err_pred_T021 = torch.eye(4, device=dev).unsqueeze(0).repeat(bs, 1, 1)
        min_err_pred_T021[:, :3, :3] = R
        min_err_pred_T021[:, :3, 3] = t.squeeze(-1) * t_scale
    # check for positive depth
    P0 = torch.eye(4, device=dev).unsqueeze(0).repeat(bs, 1, 1)[:, :3, :]
    pts_3d = triangulate_points(P0, min_err_pred_T021[:, :3, :], kpts0_norm, kpts1_norm)
    depth0 = pts_3d[..., -1]
    depth1 = depth_from_point(min_err_pred_T021[:, :3, :3], min_err_pred_T021[:, :3, 3:], pts_3d)
    pos_depth_mask = torch.logical_and(depth0 > 0., depth1 > 0.)
    epi_err = None
    inliers = None
    if determine_inliers:
        epi_err = symmetrical_epipolar_distance(kpts0_norm, kpts1_norm, Fs)
        epi_err_sqrt = epi_err.sqrt()
        thresh = 3. / ((intr0[:, 0, 0] + intr0[:, 1, 1] + intr1[:, 0, 0] + intr1[:, 1, 1]) / 4.)
        inliers = torch.logical_and(pos_depth_mask, epi_err_sqrt <= thresh.unsqueeze(-1))
    info = {"kpts0_norm": kpts0_norm, "kpts1_norm": kpts1_norm, "confidence": confidence, "inliers": inliers, \
        "pos_depth_mask": pos_depth_mask}
    return min_err_pred_T021, info

def compute_rotation_error(T0, T1, reduce=True):
    # use diagonal and sum to compute trace of a batch of matrices
    cos_a = ((T0[..., :3, :3].transpose(-1, -2) @ T1[..., :3, :3]).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) \
        - 1.) / 2.
    cos_a = torch.clamp(cos_a, -1., 1.) # avoid nan
    abs_acos_a = torch.abs(torch.arccos(cos_a))
    if reduce:
        return abs_acos_a.mean()
    else:
        return abs_acos_a

def compute_translation_error_as_angle(T0, T1, reduce=True):
    n = torch.linalg.norm(T0[..., :3, 3], dim=-1) * torch.linalg.norm(T1[..., :3, 3], dim=-1)
    valid_n = n > 1e-6
    T0_dot_T1 = (T0[..., :3, 3][valid_n] * T1[..., :3, 3][valid_n]).sum(-1)
    err = torch.abs(torch.arccos((T0_dot_T1 / n[valid_n]).clamp(-1., 1.)))
    if reduce:
        return err.mean()
    else:
        return err