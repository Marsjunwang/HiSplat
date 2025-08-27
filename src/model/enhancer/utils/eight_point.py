import torch

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
    Aw = A * torch.sqrt(w.clamp_min(1e-8)).unsqueeze(-1)
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